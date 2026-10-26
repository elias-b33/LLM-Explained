"""
Write a model (or a train.py checkpoint) as a Workshop weights file (.bin).
Format: see README.md in this folder and src/lab/engine/weights.ts.

  python export.py --ckpt C:/Users/elias/llm-pretrain/runs/xl/ckpt.pt --out ../../public/models/xl.bin
"""
from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np
import torch

from model import GPT, GPTConfig

MAGIC = b'LLMXWGT\x00'
VERSION = 1


def export(model: GPT, path: str | Path, size: str, dtype: str = 'f16', training: dict | None = None) -> int:
    """Write `model` to `path`; returns the file size in bytes."""
    assert dtype in ('f16', 'f32')
    named = model.named_tensors()
    header = {
        'format': 'llm-explainer-gpt',
        'size': size,
        'config': model.cfg.to_dict(),
        'vocab': {'type': 'bytes', 'size': model.cfg.vocabSize},
        'dtype': dtype,
        'layout': [{'name': n, 'shape': list(t.shape)} for n, t in named],
        'numParams': sum(t.numel() for _, t in named),
        'training': training or {},
    }
    js = json.dumps(header, separators=(',', ':'), ensure_ascii=False).encode('utf-8')
    js += b' ' * ((-(16 + len(js))) % 16)  # data starts 16-byte aligned
    np_dtype = np.dtype('<f2') if dtype == 'f16' else np.dtype('<f4')  # little-endian
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        f.write(MAGIC)
        f.write(struct.pack('<II', VERSION, len(js)))
        f.write(js)
        for _, t in named:
            a = t.detach().float().cpu().contiguous().numpy()  # row-major, the engine's layout
            if dtype == 'f16' and np.abs(a).max() > 65504:
                raise ValueError('weight too large for fp16')
            f.write(a.astype(np_dtype).tobytes())
    return path.stat().st_size


def load_checkpoint(ckpt_path: str | Path) -> tuple[GPT, dict]:
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model = GPT(GPTConfig(**ck['config']))
    model.load_state_dict(ck['model'])
    return model, ck


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--dtype', default='f16', choices=['f16', 'f32'])
    args = ap.parse_args()
    model, ck = load_checkpoint(args.ckpt)
    n = export(model, args.out, ck['size'], args.dtype, ck.get('training', {}))
    print(f'wrote {args.out}: {n / 1e6:.1f} MB, {model.num_params():,} params ({args.dtype})')


if __name__ == '__main__':
    main()
