"""Run TAIR's train.py unchanged, with the two shims infer_plates.py uses (plat-nomor 11-13, A5):
torch.Tuple (removed from newer torch, repo annotations still use it), and TESTR's multi-scale deformable attention
in pure PyTorch, because the compiled testr.adet._C op is not built in this env (setup: docs/SERVER_SETUP.md section 7).
The fallback (ms_deform_attn_core_pytorch, grid_sample) is differentiable, so training works, only slower.

    cd TAIR && PYTHONPATH=.:detectron2:testr .venv/bin/accelerate launch train_plates.py \\
        --config configs/train/train_plates_stage3_smoke.yaml --config_testr testr/configs/TESTR/TESTR_R_50_Polygon.yaml
"""
import runpy
import sys
import types
import typing

import torch

torch.Tuple = typing.Tuple
sys.modules['testr.adet._C'] = types.ModuleType('_C')  # skip the CUDA build; op patched to pure pytorch below
from testr.adet.layers import ms_deform_attn as msda  # noqa: E402

msda._MSDeformAttnFunction.apply = lambda v, shapes, _lsi, loc, w, _step: msda.ms_deform_attn_core_pytorch(v, shapes, loc, w)
sys.argv[0] = 'train.py'
runpy.run_path('train.py', run_name='__main__')
