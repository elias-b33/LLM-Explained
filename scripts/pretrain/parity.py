"""
Parity test, the gate for pretraining (B2): the PyTorch twin and the Workshop's
TypeScript CPU engine must compute the same function from the same .bin file.

For each case: build a GPT with random (deliberately non-trivial) weights in
PyTorch, export it, run the TS engine on the same token ids through Node
(parity_engine.ts), and compare logits and loss.

  python parity.py                 # fp32 export: max |logit diff| <= 1e-4, rel loss diff <= 1e-5
  python parity.py --keep DIR      # also keep the .bin files (e.g. to feed the in-browser GPU self-test)

The f16 case checks the real shipping format: PyTorch runs on the fp16-rounded
weights, so the comparison isolates the loader's fp16 decoding.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

from export import export
from model import GPT, GPTConfig, SIZES, _key

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
LOGIT_TOL = 1e-4
LOSS_TOL = 1e-5

CASES = [
    # odd shapes: head split, sizes that are not multiples of anything nice
    dict(name='odd', cfg=GPTConfig(256, 40, 2, 3, 48), B=3, T=40, dtype='f32'),
    dict(name='large', cfg=SIZES['large'], B=2, T=128, dtype='f32'),
    dict(name='xl', cfg=SIZES['xl'], B=1, T=256, dtype='f32'),
    dict(name='large-f16', cfg=SIZES['large'], B=2, T=128, dtype='f16'),
]


def find_node() -> str:
    for c in (shutil.which('node'), r'C:\Program Files\nodejs\node.exe'):
        if c and Path(c).exists():
            return c
    raise SystemExit('node not found')


@torch.no_grad()
def randomize(model: GPT, gen: torch.Generator):
    """Engine init plus noise on every tensor, so gains, biases and the residual stream all matter."""
    model.reset_parameters(gen)
    for name, shape, init, std, _ in model.specs:
        t = model.p[_key(name)]
        if init == 'ones':
            t.add_(0.2 * torch.randn(shape, generator=gen))
        elif init == 'zeros':
            t.add_(0.05 * torch.randn(shape, generator=gen))
        else:
            t.mul_(3.0)  # bigger than init: logits of a few units, sharper attention


def run_case(case: dict, node: str, tmp: Path, keep: Path | None) -> dict:
    cfg, B, T, dtype = case['cfg'], case['B'], case['T'], case['dtype']
    gen = torch.Generator().manual_seed(1234)
    model = GPT(cfg, sdpa=False).float().eval()
    randomize(model, gen)
    if dtype == 'f16':
        with torch.no_grad():
            for _, t in model.named_tensors():
                t.copy_(t.half().float())
    idx = torch.randint(0, cfg.vocabSize, (B, T), generator=gen)
    tgt = torch.randint(0, cfg.vocabSize, (B, T), generator=gen)
    with torch.no_grad():
        logits, loss = model(idx, tgt)
        model.sdpa = True
        logits_sdpa, _ = model(idx, tgt)
        model.sdpa = False

    bin_path = tmp / f'parity-{case["name"]}.bin'
    size = export(model, bin_path, f'parity-{case["name"]}', dtype, {'dataset': 'random weights (parity test)'})
    case_path = tmp / f'case-{case["name"]}.json'
    case_path.write_text(json.dumps({'B': B, 'T': T, 'idx': idx.flatten().tolist(), 'targets': tgt.flatten().tolist()}))
    out_path = tmp / f'logits-{case["name"]}.f32'
    r = subprocess.run([node, str(HERE / 'parity_engine.ts'), str(bin_path), str(case_path), str(out_path)],
                       cwd=REPO, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f'TS engine failed:\n{r.stderr}')
    ts = json.loads(r.stdout.strip().splitlines()[-1])
    ts_logits = np.fromfile(out_path, dtype='<f4').reshape(B, T, cfg.vocabSize)
    pt = logits.numpy()
    max_abs = float(np.abs(ts_logits - pt).max())
    rel_loss = abs(ts['loss'] - loss.item()) / abs(loss.item())
    if keep:
        keep.mkdir(parents=True, exist_ok=True)
        shutil.copy(bin_path, keep / bin_path.name)
    return {
        'case': case['name'], 'dtype': dtype, 'params': model.num_params(), 'file_mb': round(size / 1e6, 2),
        'logit_absmax': round(float(np.abs(pt).max()), 3),
        'max_abs_logit_diff': max_abs, 'loss_pt': loss.item(), 'loss_ts': ts['loss'], 'rel_loss_diff': rel_loss,
        'sdpa_vs_explicit_max_abs': float((logits_sdpa - logits).abs().max()),
        'ts_decoder_vs_forward_max_abs': ts['decoderVsForwardMaxAbs'],
        'ts_forward_s': round(ts['fwdMs'] / 1000, 2),
        'ok': max_abs <= LOGIT_TOL and rel_loss <= LOSS_TOL,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--keep', type=Path, default=None, help='copy the exported .bin files here')
    ap.add_argument('--only', default=None, help='comma-separated case names')
    args = ap.parse_args()
    torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))
    node = find_node()
    cases = [c for c in CASES if not args.only or c['name'] in args.only.split(',')]
    results = []
    with tempfile.TemporaryDirectory() as td:
        for c in cases:
            res = run_case(c, node, Path(td), args.keep)
            results.append(res)
            print(json.dumps(res))
    ok = all(r['ok'] for r in results)
    print(f'\nPARITY {"PASS" if ok else "FAIL"}: max |logit diff| {max(r["max_abs_logit_diff"] for r in results):.2e} '
          f'(tol {LOGIT_TOL:.0e}), max rel loss diff {max(r["rel_loss_diff"] for r in results):.2e} (tol {LOSS_TOL:.0e})')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
