"""
Pretrain a Workshop-sized GPT on raw UTF-8 bytes (vocab 256), on one CUDA GPU.

  python train.py --size xl                                  # the B2 run (data from prepare_data.py)
  python train.py --size large --data <dir> --max-steps 300  # smoke test

bf16 autocast, torch.compile when it works (on Windows it needs the
triton-windows package; otherwise it falls back to eager mode), AdamW with
linear warmup and cosine decay to 10% of the peak, gradient clipping at 1.0,
periodic validation, sample text and checkpoints. At the end the final weights
are exported as an fp16 Workshop .bin (export.py) with the training metadata.

Runs and checkpoints go to C:/Users/elias/llm-pretrain/runs/<size>/ (outside OneDrive).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from export import export
from model import GPT, SIZES

ROOT = Path(os.environ.get('LLM_PRETRAIN_ROOT', r'C:\Users\elias\llm-pretrain'))


@dataclass
class TrainConfig:
    batch: int          # sequences per micro-batch
    accum: int          # micro-batches per optimizer step
    lr: float           # peak learning rate (the Workshop's from-scratch LR for this size)
    tokens: float       # total training tokens (= bytes)
    warmup: int = 500   # steps
    min_lr_frac: float = 0.1
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    clip: float = 1.0
    eval_every: int = 500
    eval_tokens: int = 2_000_000
    ckpt_every: int = 2000
    ckpt_minutes: float = 5.0  # also checkpoint at least this often (wall clock)


# Tokens per step: large 32k, huge 32k, xl 64k. Budgets are several times the
# Chinchilla 20 tokens/param, since small byte-level models keep improving and
# the runs are short; the ~420 MB dataset is repeated a few times for XL.
CONFIGS: dict[str, TrainConfig] = {
    'large': TrainConfig(batch=256, accum=1, lr=2e-3, tokens=0.5e9),
    'huge': TrainConfig(batch=128, accum=1, lr=1e-3, tokens=1.0e9),
    'xl': TrainConfig(batch=128, accum=2, lr=6e-4, tokens=1.5e9),
}


class Bytes:
    def __init__(self, path: Path):
        self.data = np.memmap(path, dtype=np.uint8, mode='r')

    def __len__(self):
        return len(self.data)

    def batch(self, B: int, T: int, gen: np.random.Generator, device: str):
        ix = gen.integers(0, len(self.data) - T - 1, size=B)
        x = np.stack([self.data[i:i + T] for i in ix]).astype(np.int64)
        y = np.stack([self.data[i + 1:i + 1 + T] for i in ix]).astype(np.int64)
        return (torch.from_numpy(x).pin_memory().to(device, non_blocking=True),
                torch.from_numpy(y).pin_memory().to(device, non_blocking=True))


def lr_at(step: int, total: int, c: TrainConfig) -> float:
    if step < c.warmup:
        return c.lr * (step + 1) / c.warmup
    p = min(1.0, (step - c.warmup) / max(1, total - c.warmup))
    return c.lr * (c.min_lr_frac + (1 - c.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * p)))


@torch.no_grad()
def evaluate(model, val: Bytes, T: int, B: int, n_tokens: int, device: str) -> float:
    """Mean loss on evenly spaced windows of the held-out books (same windows every time)."""
    model.eval()
    n_win = max(B, n_tokens // T)
    starts = np.linspace(0, len(val) - T - 2, n_win).astype(np.int64)
    tot, cnt = 0.0, 0
    for k in range(0, n_win, B):
        s = starts[k:k + B]
        x = torch.from_numpy(np.stack([val.data[i:i + T] for i in s]).astype(np.int64)).to(device)
        y = torch.from_numpy(np.stack([val.data[i + 1:i + 1 + T] for i in s]).astype(np.int64)).to(device)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            _, loss = model(x, y)
        tot += loss.item() * len(s)
        cnt += len(s)
    model.train()
    return tot / cnt


def sample(raw: GPT, prompt: str, n: int, device: str) -> str:
    idx = torch.tensor([list(prompt.encode('utf-8'))], device=device)
    g = torch.Generator(device=device).manual_seed(4242)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        out = raw.generate(idx, n, 0.8, g)
    return bytes(out[0].tolist()).decode('utf-8', errors='replace')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--size', required=True, choices=sorted(SIZES))
    ap.add_argument('--data', type=Path, default=ROOT / 'data' / 'pg-en-sv')
    ap.add_argument('--out', type=Path, default=None, help='run directory (default ROOT/runs/<size>)')
    ap.add_argument('--max-steps', type=int, default=None, help='override the token budget (smoke tests)')
    ap.add_argument('--warmup', type=int, default=None)
    ap.add_argument('--eval-every', type=int, default=None)
    ap.add_argument('--batch', type=int, default=None)
    ap.add_argument('--accum', type=int, default=None, help='micro-batches per step (lower --batch, raise --accum: same tokens/step, less VRAM)')
    ap.add_argument('--no-compile', action='store_true')
    ap.add_argument('--resume', action='store_true', help='continue from <out>/ckpt.pt')
    ap.add_argument('--export', type=Path, default=None, help='also write the final fp16 .bin here')
    ap.add_argument('--seed', type=int, default=1337)
    args = ap.parse_args()

    assert torch.cuda.is_available(), 'needs a CUDA GPU'
    device = 'cuda'
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    mcfg = SIZES[args.size]
    tc = CONFIGS[args.size]
    if args.batch:
        tc.batch = args.batch
    if args.accum:
        tc.accum = args.accum
    if args.warmup is not None:
        tc.warmup = args.warmup
    if args.eval_every:
        tc.eval_every = args.eval_every
    T = mcfg.ctx
    tok_per_step = tc.batch * tc.accum * T
    total = args.max_steps or int(tc.tokens // tok_per_step)
    tc.warmup = min(tc.warmup, max(1, total // 10))
    out = args.out or ROOT / 'runs' / args.size
    out.mkdir(parents=True, exist_ok=True)

    train, val = Bytes(args.data / 'train.bin'), Bytes(args.data / 'val.bin')
    meta_path = args.data / 'meta.json'
    dataset = json.loads(meta_path.read_text(encoding='utf-8')) if meta_path.exists() else {}
    ds_name = f'Project Gutenberg en+sv ({len(train) / 1e6:.0f} MB, {args.data.name})'

    raw = GPT(mcfg).to(device)
    opt = torch.optim.AdamW(raw.optim_groups(tc.weight_decay), lr=tc.lr, betas=tc.betas, eps=1e-8, fused=True)
    step, tokens_seen, best_val = 0, 0, float('inf')
    if args.resume and (out / 'ckpt.pt').exists():
        ck = torch.load(out / 'ckpt.pt', map_location=device, weights_only=False)
        raw.load_state_dict(ck['model'])
        opt.load_state_dict(ck['opt'])
        step, tokens_seen = ck['step'], ck['training']['tokensSeen']
        resumed_val = ck['training'].get('finalValLoss', float('nan'))
        print(f'resumed at step {step} ({tokens_seen / 1e6:.0f}M tokens)')
    else:
        resumed_val = float('nan')

    model = raw
    compiled = False
    if not args.no_compile:
        try:
            model = torch.compile(raw)
            x, y = train.batch(2, T, np.random.default_rng(0), device)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                _, l = model(x, y)
            l.backward()
            opt.zero_grad(set_to_none=True)
            compiled = True
        except Exception as e:  # noqa: BLE001 - any compiler/toolchain failure: run eagerly
            print(f'torch.compile unavailable ({type(e).__name__}: {str(e).splitlines()[0][:160]}); using eager mode')
            model = raw
            opt.zero_grad(set_to_none=True)

    print(f'{args.size}: {raw.num_params():,} params, ctx {T}, {tok_per_step:,} tokens/step, {total:,} steps '
          f'({total * tok_per_step / 1e9:.2f}B tokens), train {len(train) / 1e6:.0f} MB, val {len(val) / 1e6:.1f} MB, '
          f'compile={compiled}')
    log = open(out / 'log.jsonl', 'a', encoding='utf-8')
    gen = np.random.default_rng(args.seed + step)
    prompts = ['Once upon a time ', 'Det var en gång ']
    last_val = resumed_val
    last_ckpt = time.time()
    t_last, tok_last = time.time(), tokens_seen
    train_loss = float('nan')

    def save(tag: str = 'ckpt'):
        # write to a temp file and swap it in, so a crash mid-save never destroys the last good checkpoint
        tmp = out / f'{tag}.pt.tmp'
        torch.save({
            'model': raw.state_dict(), 'opt': opt.state_dict(), 'step': step, 'size': args.size,
            'config': mcfg.to_dict(), 'train_config': tc.__dict__,
            'training': training_meta(),
        }, tmp)
        os.replace(tmp, out / f'{tag}.pt')

    def training_meta():
        return {
            'dataset': ds_name, 'datasetSource': dataset.get('source'), 'tokensSeen': tokens_seen, 'steps': step,
            'finalValLoss': last_val, 'finalTrainLoss': train_loss, 'peakLr': tc.lr,
            'date': datetime.now(timezone.utc).isoformat(timespec='seconds'),
            'trainer': 'scripts/pretrain/train.py (PyTorch, bf16 autocast)',
        }

    while step < total:
        lr = lr_at(step, total, tc)
        for g in opt.param_groups:
            g['lr'] = lr
        acc = 0.0
        for _ in range(tc.accum):
            x, y = train.batch(tc.batch, T, gen, device)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                _, loss = model(x, y)
            (loss / tc.accum).backward()
            acc += loss.detach()
        gnorm = torch.nn.utils.clip_grad_norm_(raw.parameters(), tc.clip)
        opt.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        tokens_seen += tok_per_step

        if step % 50 == 0 or step == total:
            train_loss = (acc / tc.accum).item()
            now = time.time()
            tps = (tokens_seen - tok_last) / (now - t_last)
            t_last, tok_last = now, tokens_seen
            eta = (total - step) * tok_per_step / max(tps, 1)
            print(f'step {step}/{total} loss {train_loss:.4f} lr {lr:.2e} gnorm {gnorm.item():.2f} '
                  f'{tps / 1e3:.0f}k tok/s eta {eta / 60:.1f} min', flush=True)
            log.write(json.dumps({'step': step, 'loss': train_loss, 'lr': lr, 'tok_s': tps, 'tokens': tokens_seen}) + '\n')
        if step % tc.eval_every == 0 or step == total:
            last_val = evaluate(model, val, T, tc.batch, tc.eval_tokens, device)
            txts = [sample(raw, p, 160, device) for p in prompts]
            print(f'  val {last_val:.4f} | ' + ' | '.join(repr(t) for t in txts), flush=True)
            log.write(json.dumps({'step': step, 'val': last_val, 'samples': txts}, ensure_ascii=False) + '\n')
            log.flush()
            if last_val < best_val:
                best_val = last_val
        if step % tc.ckpt_every == 0 or step == total or time.time() - last_ckpt > tc.ckpt_minutes * 60:
            save()
            last_ckpt = time.time()

    meta = training_meta()
    (out / 'final.json').write_text(json.dumps(meta, indent=1), encoding='utf-8')
    if args.export:
        n = export(raw, args.export, args.size, 'f16', meta)
        print(f'exported {args.export} ({n / 1e6:.1f} MB)')
    print(json.dumps(meta, indent=1))


if __name__ == '__main__':
    main()
