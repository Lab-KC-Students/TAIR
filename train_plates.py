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
    python3 train_plates.py --summary model_ckpts/plates/stage3_smoke/train_log.jsonl   # losses + smoke checks, stdlib
"""
import importlib.machinery
import importlib.util
import json
import os
import runpy
import sys
import time
import types
import typing


def say(s):
    print(s, flush=True)  # stdout into `| tee` is block-buffered: unflushed lines are lost if the run is killed


def local_wandb(exp_dir, out=say):
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
    # importlib.util.find_spec('wandb') (accelerate's is_wandb_available, run when train.py imports accelerate) raises
    # ValueError on a sys.modules entry whose __spec__ is None, which a bare ModuleType has (11-15 smoke crash)
    w.__spec__ = importlib.machinery.ModuleSpec('wandb', None)
    w.log = log
    w.login = w.init = w.finish = w.watch = lambda *a, **k: None
    w.Image = lambda *a, **k: None
    w.log_path = path
    write({'run_start': time.strftime('%Y-%m-%d %H:%M:%S'), 'argv': sys.argv})
    return w


SMOKE_MAX_GB = 48  # L40


def summary(path, max_gb=SMOKE_MAX_GB, out=say, max_rows=25):
    """Read a train_log.jsonl (last run only): loss table per log step, s/step, peak GPU memory, val metrics, and the
    smoke checks of HOW_TO_RUN.md 5g step 5. Returns (passed, reasons). Stdlib only. A long run (10k steps = 200 logs)
    prints ~20 evenly spaced rows + the last one; the checks and s/step use every row."""
    recs = [json.loads(ln) for ln in open(path) if ln.strip()]
    starts = [i for i, r in enumerate(recs) if 'run_start' in r]
    run = recs[starts[-1] + 1:] if starts else recs
    by_step = {}
    for r in run:
        if any(k.startswith('train_loss/') for k in r):
            by_step.setdefault(r['step'], {}).update(r)
    rows = [by_step[s] for s in sorted(by_step) if 'train_loss/total_step_loss' in by_step[s]]
    out(f'# {path}: run started {recs[starts[-1]]["run_start"] if starts else "?"}, {len(run)} records')
    val = [r for r in run if any(k.startswith('sampling_val_METRIC/tot_') for k in r)]
    for r in val:
        out(f"val at step {r['step']}: " + ', '.join(f"{k.split('tot_val_')[-1]} {v:.3f}" for k, v in r.items()
                                                     if k.startswith('sampling_val_METRIC/tot_')))
    out('| step | total | diffusion | ocr | s/step | peak GB |')
    out('|---|---|---|---|---|---|')
    sps = []
    every = 1 if len(rows) <= max_rows else -(-len(rows) // 20)
    for i, r in enumerate(rows):
        s = None
        if i:
            s = (r['time_s'] - rows[i - 1]['time_s']) / (r['step'] - rows[i - 1]['step'])
            sps.append(s)
        if i % every and i != len(rows) - 1:
            continue
        out(f"| {r['step']} | {r['train_loss/total_step_loss']:.4f} | {r.get('train_loss/diffusion_loss', float('nan')):.4f} | "
            f"{r.get('train_loss/ocr_tot_loss', float('nan')):.3f} | {'–' if s is None else f'{s:.2f}'} | {r.get('peak_mem_gb')} |")
    reasons = []
    if len(rows) < 10:
        reasons.append(f'only {len(rows)} loss logs (last step {rows[-1]["step"] if rows else "none"}): the run stopped '
                       'early or was still running when the log was copied')
    vals = [v for r in rows for k, v in r.items() if k.startswith('train_loss') and isinstance(v, float)]
    if any(v != v or v in (float('inf'), float('-inf')) for v in vals):
        reasons.append('nan / inf in the losses')
    if len(rows) >= 10:
        first, last = rows[:5], rows[-5:]

        def m(rs, k):
            return sum(r.get(k, 0.0) for r in rs) / len(rs)
        if not m(last, 'train_loss/ocr_tot_loss') < m(first, 'train_loss/ocr_tot_loss'):
            reasons.append('OCR loss did not fall (mean of the last 5 logs >= first 5)')
        if m(last, 'train_loss/diffusion_loss') > 1.1 * m(first, 'train_loss/diffusion_loss'):
            reasons.append('diffusion loss climbed (last 5 > 1.1 x first 5)')
        out(f"OCR loss first 5 -> last 5: {m(first, 'train_loss/ocr_tot_loss'):.3f} -> {m(last, 'train_loss/ocr_tot_loss'):.3f}; "
            f"diffusion {m(first, 'train_loss/diffusion_loss'):.4f} -> {m(last, 'train_loss/diffusion_loss'):.4f}")
    peak = max((r.get('peak_mem_gb') or 0 for r in run), default=0)
    if peak >= max_gb:
        reasons.append(f'peak GPU memory {peak} GB >= {max_gb} GB')
    steady = sorted(sps[1:] or sps)  # the first interval includes the step-1 val pass
    if steady:
        med = steady[len(steady) // 2]
        out(f'median {med:.2f} s/step -> 10k steps ~ {med * 1e4 / 3600:.1f} h of training (+ val passes); peak {peak} GB')
    out('smoke: PASS' if not reasons else 'smoke: FAIL: ' + '; '.join(reasons))
    return not reasons, reasons


def _selftest_summary():
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, 'train_log.jsonl')

        def write(n, ocr0, nan=False):
            with open(p, 'w') as f:
                f.write(json.dumps({'run_start': 'old'}) + '\n' + json.dumps({'step': 10, 'time_s': 1,
                                                                              'train_loss/total_step_loss': 9.9}) + '\n')
                f.write(json.dumps({'run_start': 'new'}) + '\n')
                f.write(json.dumps({'step': 1, 'time_s': 60, 'sampling_val_METRIC/tot_val_psnr': 19.0}) + '\n')
                for i in range(1, n + 1):
                    ocr = float('nan') if nan and i == n else ocr0 - i
                    for k, v in (('train_loss/diffusion_loss', 0.2), ('train_loss/ocr_tot_loss', ocr),
                                 ('train_loss/total_step_loss', 0.2 + 0.05 * ocr)):
                        f.write(json.dumps({'step': 10 * i, 'time_s': 60 + 20 * i, 'peak_mem_gb': 28.8, k: v}) + '\n')
        quiet = []
        write(20, 60)
        assert summary(p, out=quiet.append) == (True, []), quiet
        assert any('2.00 s/step' in s for s in quiet), quiet
        write(1, 60)
        ok, why = summary(p, out=quiet.append)
        assert not ok and 'only 1 loss logs (last step 10)' in why[0], why  # the 11-15 copy: stopped at step 10
        write(20, 60, nan=True)
        assert not summary(p, out=quiet.append)[0]
        write(200, 300)  # a 10k-step run at log_loss_every 50: thinned table, last step kept
        quiet = []
        assert summary(p, out=quiet.append) == (True, []), quiet
        table = [s for s in quiet if s.startswith('| ') and s[2].isdigit()]
        assert 15 <= len(table) <= 22 and table[-1].startswith('| 2000 |'), table


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
    # installed as sys.modules['wandb'] (as below), find_spec must work: accelerate calls it on import
    saved = sys.modules.get('wandb')
    sys.modules['wandb'] = w
    try:
        spec = importlib.util.find_spec('wandb')
        assert spec is not None and spec.name == 'wandb', spec
    finally:
        if saved is None:
            del sys.modules['wandb']
        else:
            sys.modules['wandb'] = saved
    print('ok: train_plates self-test')


if __name__ == '__main__':
    if '--selftest' in sys.argv:
        _selftest()
        _selftest_summary()
        print('ok: train_plates summary self-test')
        sys.exit()
    if '--summary' in sys.argv:
        sys.exit(0 if summary(sys.argv[sys.argv.index('--summary') + 1])[0] else 1)
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
