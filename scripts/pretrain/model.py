"""
PyTorch twin of the Workshop's TypeScript GPT (src/lab/engine/gpt.ts).

Everything that affects the numbers is replicated exactly:
  * x = tok[idx] + pos[t]
  * per block: x = x + Wo(attn(LN1(x))) ; x = x + Wproj(gelu_tanh(Wfc(LN2(x))))
  * logits = LNf(x) @ wout + bout     (separate, untied LM head)
  * LayerNorm eps 1e-5 (biased variance), tanh-GELU with sqrt(2/pi) and 0.044715,
    causal softmax attention scaled by 1/sqrt(head_size)
  * qkv rows are [q | k | v], each d wide; head h uses columns h*hs:(h+1)*hs
  * weights are stored as [in, out] and applied as x @ W + b (NOT nn.Linear's
    [out, in]), so every tensor has the engine's shape and memory layout and
    export is a plain concatenation
  * init: N(0, 0.02) for matrices and tok; N(0, 0.01) for pos; N(0, 0.02/sqrt(2L))
    for the two residual projections (wo, wproj); biases 0, LN gains 1
  * weight decay on matrices and embeddings only (tok, pos, wqkv, wo, wfc, wproj, wout)

Parameter names and order equal GPT.layout() in the TS engine.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

LN_EPS = 1e-5


@dataclass
class GPTConfig:
    vocabSize: int = 256
    ctx: int = 256
    nLayer: int = 8
    nHead: int = 8
    dModel: int = 512

    def to_dict(self) -> dict:
        return asdict(self)


# Workshop sizes that can be pretrained (must match src/lab/sizes.ts; vocab is the 256 bytes)
SIZES: dict[str, GPTConfig] = {
    'large': GPTConfig(256, 128, 4, 6, 192),
    'huge': GPTConfig(256, 256, 6, 6, 384),
    'xl': GPTConfig(256, 256, 8, 8, 512),
}


def layout(cfg: GPTConfig) -> list[tuple[str, tuple[int, ...], str, float, bool]]:
    """(name, shape, init, std, decay) for every tensor, in the engine's storage order."""
    V, T, L, d = cfg.vocabSize, cfg.ctx, cfg.nLayer, cfg.dModel
    std = 0.02
    proj = 0.02 / math.sqrt(2 * L)
    out = [('tok', (V, d), 'normal', std, True), ('pos', (T, d), 'normal', 0.01, True)]
    for l in range(L):
        out += [
            (f'{l}.ln1g', (d,), 'ones', 0, False),
            (f'{l}.ln1b', (d,), 'zeros', 0, False),
            (f'{l}.wqkv', (d, 3 * d), 'normal', std, True),
            (f'{l}.bqkv', (3 * d,), 'zeros', 0, False),
            (f'{l}.wo', (d, d), 'normal', proj, True),
            (f'{l}.bo', (d,), 'zeros', 0, False),
            (f'{l}.ln2g', (d,), 'ones', 0, False),
            (f'{l}.ln2b', (d,), 'zeros', 0, False),
            (f'{l}.wfc', (d, 4 * d), 'normal', std, True),
            (f'{l}.bfc', (4 * d,), 'zeros', 0, False),
            (f'{l}.wproj', (4 * d, d), 'normal', proj, True),
            (f'{l}.bproj', (d,), 'zeros', 0, False),
        ]
    out += [
        ('lnfg', (d,), 'ones', 0, False),
        ('lnfb', (d,), 'zeros', 0, False),
        ('wout', (d, V), 'normal', std, True),
        ('bout', (V,), 'zeros', 0, False),
    ]
    return out


def _key(name: str) -> str:
    # nn.ParameterDict keys may not contain '.'
    return name.replace('.', '_')


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig, sdpa: bool = True):
        super().__init__()
        assert cfg.dModel % cfg.nHead == 0
        self.cfg = cfg
        # sdpa=True uses PyTorch's fused (flash / memory-efficient) attention kernel: the same maths,
        # just faster. sdpa=False spells out softmax(QK^T/sqrt(hs)) V, op for op like the TS engine.
        self.sdpa = sdpa
        self.specs = layout(cfg)
        self.p = nn.ParameterDict({_key(n): nn.Parameter(torch.empty(shape)) for n, shape, *_ in self.specs})
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self, generator: torch.Generator | None = None):
        for name, _shape, init, std, _decay in self.specs:
            t = self.p[_key(name)]
            if init == 'ones':
                t.fill_(1.0)
            elif init == 'zeros':
                t.zero_()
            else:
                t.normal_(0.0, std, generator=generator)

    def named_tensors(self) -> list[tuple[str, torch.Tensor]]:
        """Parameters in the engine's storage order."""
        return [(n, self.p[_key(n)]) for n, *_ in self.specs]

    def num_params(self) -> int:
        return sum(t.numel() for _, t in self.named_tensors())

    def optim_groups(self, weight_decay: float):
        decay = [self.p[_key(n)] for n, _s, _i, _std, dec in self.specs if dec]
        no_decay = [self.p[_key(n)] for n, _s, _i, _std, dec in self.specs if not dec]
        return [{'params': decay, 'weight_decay': weight_decay}, {'params': no_decay, 'weight_decay': 0.0}]

    def _attn(self, qkv: torch.Tensor) -> torch.Tensor:
        B, T, _ = qkv.shape
        H, d = self.cfg.nHead, self.cfg.dModel
        hs = d // H
        q, k, v = qkv.split(d, dim=2)
        q = q.view(B, T, H, hs).transpose(1, 2)
        k = k.view(B, T, H, hs).transpose(1, 2)
        v = v.view(B, T, H, hs).transpose(1, 2)
        if self.sdpa:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=1.0 / math.sqrt(hs))
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(hs))
            mask = torch.ones(T, T, dtype=torch.bool, device=qkv.device).tril()
            att = att.masked_fill(~mask, float('-inf'))
            att = torch.softmax(att, dim=-1)
            y = att @ v
        return y.transpose(1, 2).contiguous().view(B, T, d)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        """idx, targets: [B, T] int64. Returns (logits [B, T, V], mean cross-entropy or None)."""
        cfg, p = self.cfg, self.p
        B, T = idx.shape
        assert T <= cfg.ctx, 'sequence longer than context'
        d = cfg.dModel
        x = p['tok'][idx] + p['pos'][:T]
        for l in range(cfg.nLayer):
            g = lambda n: p[f'{l}_{n}']  # noqa: E731
            h = F.layer_norm(x, (d,), g('ln1g'), g('ln1b'), LN_EPS)
            qkv = h @ g('wqkv') + g('bqkv')
            x = x + (self._attn(qkv) @ g('wo') + g('bo'))
            h = F.layer_norm(x, (d,), g('ln2g'), g('ln2b'), LN_EPS)
            h = F.gelu(h @ g('wfc') + g('bfc'), approximate='tanh')
            x = x + (h @ g('wproj') + g('bproj'))
        x = F.layer_norm(x, (d,), p['lnfg'], p['lnfb'], LN_EPS)
        logits = x @ p['wout'] + p['bout']
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.float().view(-1, cfg.vocabSize), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, n: int, temperature: float = 0.8, generator: torch.Generator | None = None):
        """Plain (uncached) sampling, for eyeballing checkpoints."""
        for _ in range(n):
            logits, _ = self(idx[:, -self.cfg.ctx:])
            probs = torch.softmax(logits[:, -1].float() / max(temperature, 1e-3), dim=-1)
            nxt = torch.multinomial(probs, 1, generator=generator)
            idx = torch.cat([idx, nxt], dim=1)
        return idx
