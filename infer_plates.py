# TeReDiff inference on LQ-only images (val.py needs an HQ twin per image and computes IQA metrics; this doesn't).
# usage (from TAIR/): PYTHONPATH=.:detectron2:testr .venv/bin/python infer_plates.py <img_dir> <out_dir> [label_file]
# writes <out_dir>/restored_<name>.png and <out_dir>/predictions.csv (last-timestep TESTR text per image)
import os, re, sys, csv, types, typing
import torch
torch.Tuple = typing.Tuple  # removed from newer torch, repo annotations still use it
sys.modules['testr.adet._C'] = types.ModuleType('_C')  # skip the CUDA build; op patched to pure pytorch below

from PIL import Image
from natsort import natsorted
from omegaconf import OmegaConf
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from terediff.utils.common import instantiate_from_config
from terediff.sampler import SpacedSampler
from testr.adet.layers import ms_deform_attn as msda
from testr.adet.modeling.transformer_detector import TransformerDetector
from testr.adet.config import get_cfg

# ponytail: pure-pytorch deformable attn is slower than the compiled op, fine for 100 images
msda._MSDeformAttnFunction.apply = lambda v, shapes, _lsi, loc, w, _step: msda.ms_deform_attn_core_pytorch(v, shapes, loc, w)

STAGE3, MERGED = 'weights/terediff_stage3.pt', 'weights/terediff_fp16.pt'
img_dir, out_dir = sys.argv[1], sys.argv[2]
label_file = sys.argv[3] if len(sys.argv) > 3 else None
dev = 'cuda'
cfg = OmegaConf.load('configs/val/val_terediff.yaml')
cfg.exp_args.mode = 'VAL'

tcfg = get_cfg()
tcfg.merge_from_file('testr/configs/TESTR/TESTR_R_50_Polygon.yaml')
tcfg.freeze()

with torch.device('meta'):  # no RAM for the 6.7GB fp32 init; weights are assigned straight from the mmap'd ckpt
    cldm = instantiate_from_config(cfg.model.cldm)
swinir = instantiate_from_config(cfg.model.swinir)
detector = TransformerDetector(tcfg)

src = MERGED if os.path.exists(MERGED) else STAGE3
ckpt = torch.load(src, map_location='cpu', mmap=True, weights_only=False)
print(f'loading {src}: {list(ckpt)}')
for name, m in [('cldm', cldm), ('swinir', swinir), ('testr', detector)]:
    missing, unexpected = m.load_state_dict(ckpt[name], strict=False, assign=True)
    print(f'{name}: missing {len(missing)} unexpected {len(unexpected)} {missing[:5]}')
    assert name == 'testr' or not missing, f'{name} has missing weights, stage3 ckpt alone is not enough'
n = cldm.clip.model.context_length  # persistent=False buffer, absent from the ckpt by design, stuck on meta otherwise
cldm.clip.model.attn_mask = torch.empty(n, n).fill_(float('-inf')).triu_(1)  # same causal mask CLIP.__init__ builds

cldm.cast_dtype(torch.float16)  # unet+controlnet blocks fp16 (GroupNorm stays fp32), vae/clip fp32 -> fits 8GB
cldm, swinir, detector = cldm.to(dev).eval(), swinir.to(dev).eval(), detector.to(dev).eval()
del ckpt  # .to(dev) copied everything off the mmap; drop the reference so the file can actually be freed
if src == STAGE3:
    os.remove(STAGE3)  # 6.85GB fp32 -> free the disk before writing the ~4.3GB fp16 merge, not after
    torch.save({'cldm': cldm.state_dict(), 'swinir': swinir.state_dict(), 'testr': detector.state_dict()}, MERGED)
    print(f'saved {MERGED}, deleted {STAGE3}')

detector.test_score_threshold = 0.5
def fp32(x):  # fp16 unet features -> fp32 testr
    if torch.is_tensor(x): return x.float()
    if isinstance(x, dict): return {k: fp32(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return type(x)(fp32(v) for v in x)
    return x
ts_model = lambda feats, tgt, mode: detector(fp32(feats), tgt, mode)

diffusion = instantiate_from_config(cfg.model.diffusion).to(dev)
sampler = SpacedSampler(diffusion.betas, diffusion.parameterization, rescale_cfg=False)
def letterbox(img, s=512):  # val.py squashes to 512x512, which stretches ~3:1 plates 3x vertically; keep aspect, edge-pad to square
    w, h = img.size
    nw, nh = max(1, round(w * s / max(w, h))), max(1, round(h * s / max(w, h)))
    l, t = (s - nw) // 2, (s - nh) // 2
    img = TF.pad(img.resize((nw, nh), Image.BICUBIC), [l, t, s - nw - l, s - nh - t], padding_mode='edge')
    return TF.to_tensor(img), (l, t, nw, nh)
gen = torch.Generator(dev)
os.makedirs(out_dir, exist_ok=True)

names = natsorted(f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg')))
rows = []
for n, name in enumerate(names):
    gen.manual_seed(25)  # val.py seeds once; per-image seeding makes each result reproducible on its own
    lq, (l, t, nw, nh) = letterbox(Image.open(os.path.join(img_dir, name)).convert('RGB'))
    lq = lq.unsqueeze(0).to(dev)
    with torch.no_grad():
        cond = cldm.prepare_condition(swinir(lq), [''])
        z, ts_results = sampler.val_sample(
            model=cldm, device=dev, steps=50, x_size=(1, 4, 64, 64), cond=cond, uncond=None, cfg_scale=1.0,
            x_T=torch.randn((1, 4, 64, 64), generator=gen, device=dev), progress=False,
            cfg=cfg, pure_cldm=cldm, ts_model=ts_model, val_prompt=[''])
        restored = torch.clamp((cldm.vae_decode(z) + 1) / 2, 0, 1)[..., t:t + nh, l:l + nw]  # drop the padding
    TF.to_pil_image(restored.squeeze().float().cpu()).save(f'{out_dir}/restored_{os.path.splitext(name)[0]}.png')
    texts = ts_results[-1]['pred_texts']
    plate = re.sub(r'[^A-Z0-9]', '', ''.join(texts).upper())  # plates are alnum; TESTR may split one plate into words
    rows.append((name, plate))
    print(f'[{n + 1}/{len(names)}] {name}\t{plate}\t{texts}', flush=True)

with open(f'{out_dir}/predictions.csv', 'w', newline='') as f:
    csv.writer(f).writerows([('Name of File', 'Vehicleregistrationplatebymodel'), *rows])

if label_file:  # competition metric, same as satria-data/demo.py: pad to 9 with '*', per-position match
    real = [l.strip().upper() for l in open(label_file) if l.strip()]
    assert len(real) == len(rows), f'{len(real)} labels vs {len(rows)} images'
    acc = sum(sum(a == b for a, b in zip(t.ljust(9, '*'), p.ljust(9, '*'))) / 9 for t, (_, p) in zip(real, rows)) / len(rows)
    print(f'Test Accuracy : {acc}')
