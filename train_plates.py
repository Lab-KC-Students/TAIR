"""Run TAIR's train.py unchanged, with the two shims infer_plates.py uses (plat-nomor 11-13, A5):
torch.Tuple (removed from newer torch, repo annotations still use it), and TESTR's multi-scale deformable attention
in pure PyTorch, because the compiled testr.adet._C op is not built in this env (setup: docs/SERVER_SETUP.md section 7).
The fallback (ms_deform_attn_core_pytorch, grid_sample) is differentiable, so training works, only slower.

Plus a local stand-in for the wandb module (11-15): train.py reports the averaged losses and val metrics only through
wandb.log, and only when log_tool is 'wandb' (the plate configs set it). Here those numbers go to
<exp_dir>/train_log.jsonl, and one line per loss / val log is printed ("[train_log] ..."); nothing leaves the machine,
no wandb account or key is used. Images (wandb.Image) are dropped.

    cd TAIR && PYTHONPATH=.:detectron2:testr .venv/bin/accelerate launch train_plates.py \\
        --config configs/train/train_plates_stage3_smoke.yaml --config_testr testr/configs/TESTR/TESTR_R_50_Polygon.yaml
    python3 train_plates.py --selftest          # the wandb stand-in only (no GPU, no TAIR imports)
"""
import json
import os
import runpy
import sys
import time
import types
import typing


def local_wandb(exp_dir, out=print):
    """wandb stand-in. train.py logs {'global_step': g, ...} at the start of every batch (before g += 1), then the
    loss / val dicts after g += 1, so those belong to step g + 1 = the step just completed."""
    os.makedirs(exp_dir, exist_ok=True)
    path = os.path.join(exp_dir, 'train_log.jsonl')
    f = open(path, 'a')
    st = dict(g=0, t0=time.time(), t_last=time.time(), g_last=0, pending={})

    def peak_gb():
        try:
            import torch
            return round(torch.cuda.max_memory_allocated() / 2**30, 2) if torch.cuda.is_available() else None
        except ImportError:
            return None

    def write(rec):
        f.write(json.dumps(rec) + '\n')
        f.flush()

    def log(d, *_a, **_k):
        d = {k: v for k, v in d.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
        if 'global_step' in d:
            st['g'] = d['global_step']
            return
        if not d:
            return
        step, now = st['g'] + 1, time.time()
        write({'step': step, 'time_s': round(now - st['t0'], 1), 'peak_mem_gb': peak_gb(), **d})
        if any(k.startswith('train_loss') for k in d):
            st['pending'].update(d)
        if 'train_loss/total_step_loss' in d:
            p = st['pending']
            sps = (now - st['t_last']) / max(1, step - st['g_last'])
            ocr = ' '.join(f"{k.split('/')[-1]} {v:.3f}" for k, v in sorted(p.items()) if 'ocr_components' in k)
            out(f"[train_log] step {step} total {p['train_loss/total_step_loss']:.4f} "
                f"diffusion {p.get('train_loss/diffusion_loss', float('nan')):.4f} "
                f"ocr {p.get('train_loss/ocr_tot_loss', float('nan')):.4f} | {sps:.2f} s/step | "
                f"peak {peak_gb()} GB | {ocr}")
            st.update(t_last=now, g_last=step, pending={})
        elif any(k.startswith('sampling_val_METRIC/tot_') for k in d):
            out(f"[train_log] val at step {step}: " + ' '.join(f"{k.split('tot_val_')[-1]} {v:.4f}" for k, v in d.items()))

    w = types.ModuleType('wandb')
    w.log = log
    w.login = w.init = w.finish = w.watch = lambda *a, **k: None
    w.Image = lambda *a, **k: None
    w.log_path = path
    write({'run_start': time.strftime('%Y-%m-%d %H:%M:%S'), 'argv': sys.argv})
    return w


def _selftest():
    import tempfile
    lines = []
    with tempfile.TemporaryDirectory() as d:
        w = local_wandb(d, lines.append)
        for g in range(10):  # train.py order: global_step at batch start, losses after g += 1 on log steps
            w.log({'global_step': g, 'epoch': 0, 'learning_rate': 1e-5})
        w.log({'train_loss_ocr_components/loss_texts': 1.5})
        w.log({'train_loss/diffusion_loss': 0.2})
        w.log({'train_loss/ocr_tot_loss': 3.0})
        w.log({'train_loss/total_step_loss': 0.5})
        w.log({'sampling_val_VIS/x': w.Image(None)})  # image only -> nothing written
        w.log({'sampling_val_METRIC/tot_val_psnr': 21.5, 'sampling_val_METRIC/tot_val_ssim': 0.61})
        recs = [json.loads(ln) for ln in open(os.path.join(d, 'train_log.jsonl'))]
    assert 'run_start' in recs[0] and len(recs) == 6, recs
    assert recs[4]['step'] == 10 and recs[4]['train_loss/total_step_loss'] == 0.5, recs[4]
    assert lines[0].startswith('[train_log] step 10 total 0.5000 diffusion 0.2000 ocr 3.0000') and 'loss_texts 1.500' in lines[0]
    assert lines[1] == '[train_log] val at step 10: psnr 21.5000 ssim 0.6100', lines
    print('ok: train_plates self-test')


if __name__ == '__main__':
    if '--selftest' in sys.argv:
        _selftest()
        sys.exit()
    import torch
    from omegaconf import OmegaConf

    torch.Tuple = typing.Tuple
    sys.modules['testr.adet._C'] = types.ModuleType('_C')  # skip the CUDA build; op patched to pure pytorch below
    from testr.adet.layers import ms_deform_attn as msda

    msda._MSDeformAttnFunction.apply = lambda v, shapes, _lsi, loc, w, _step: msda.ms_deform_attn_core_pytorch(v, shapes, loc, w)
    cfg = OmegaConf.load(sys.argv[sys.argv.index('--config') + 1])
    assert cfg.log_args.log_tool == 'wandb', 'set log_args.log_tool: wandb (routed to the local stand-in, not wandb.ai)'
    sys.modules['wandb'] = local_wandb(cfg.train.exp_dir)
    print(f"losses + val metrics -> {sys.modules['wandb'].log_path} (local, no wandb account)")
    sys.argv[0] = 'train.py'
    runpy.run_path('train.py', run_name='__main__')
