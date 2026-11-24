"""Class-conditional tabular diffusion + dual-guided sampler.

Notation follows the paper:
  forward  q(x_t|x_0) = N(√ᾱ_t x_0, (1-ᾱ_t) I)
  CFG      ε̂ = (1+w) ε_θ(x_t,t,y) − w ε_θ(x_t,t,∅)
  Tweedie  x̂_0 = (x_t − √(1-ᾱ_t) ε̂) / √ᾱ_t
  guidance g_t = -lambda_c(t) grad V(x̂_0) + lambda_b(t) grad B(x̂_0, y)  (gradient w.r.t. x̂_0,
           dropping the Jacobian of eps_theta -- cheaper and effective at late steps), implemented as
           x̂_0' = x̂_0 + g_t ;  ε̃ = (x_t − √ᾱ_t x̂_0')/√(1−ᾱ_t) = ε̂ − √ᾱ_t g_t/√(1−ᾱ_t)
  step     x_{t−1} = (x_t − β_t/√(1-ᾱ_t) ε̃)/√α_t + σ_t z   (DDPM respaced)

Categorical/bit variables are diffused as Gaussians on {-1,+1} codes ("analog bits") and
decoded by argmax / zero threshold in the encoder.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import Z_CLIP
from .utils import LOG


# ------------------------------------------------------------- schedule
def cosine_alpha_bar(T: int, s: float = 0.008) -> np.ndarray:
    t = np.arange(T + 1) / T
    f = np.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    ab = f / f[0]
    return np.clip(ab[1:], 1e-5, 1.0)  # ᾱ_1..ᾱ_T


# ---------------------------------------------------------------- model
class TimeEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        a = t.float()[:, None] * freqs[None]
        return self.mlp(torch.cat([a.sin(), a.cos()], 1))


class ResBlock(nn.Module):
    def __init__(self, h, dropout):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(h), nn.SiLU(), nn.Linear(h, h), nn.Dropout(dropout),
                                 nn.SiLU(), nn.Linear(h, h))

    def forward(self, x):
        return x + self.net(x)


class Denoiser(nn.Module):
    def __init__(self, d, n_classes, hidden=512, depth=4, dropout=0.0):
        super().__init__()
        self.n_classes = n_classes
        self.inp = nn.Linear(d, hidden)
        self.temb = TimeEmb(hidden)
        self.cemb = nn.Embedding(n_classes + 1, hidden)  # index n_classes = null label (CFG)
        self.blocks = nn.ModuleList([ResBlock(hidden, dropout) for _ in range(depth)])
        self.out = nn.Sequential(nn.LayerNorm(hidden), nn.SiLU(), nn.Linear(hidden, d))

    def forward(self, x, t, y):
        h = self.inp(x) + self.temb(t) + self.cemb(y)
        for b in self.blocks:
            h = b(h)
        return self.out(h)


@dataclass
class DiffusionConfig:
    T: int = 1000
    hidden: int = 512
    depth: int = 4
    dropout: float = 0.0
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 1024
    steps: int = 30000
    p_uncond: float = 0.1
    ema: float = 0.999
    class_balance_power: float = 0.5  # sampling weight proportional to n_c^(-power)
    sample_steps: int = 200
    cfg_w: float = 1.0
    log_every: int = 2000


class TabDiffusion:
    def __init__(self, d: int, classes: list, cfg: DiffusionConfig, device):
        self.d, self.classes, self.cfg, self.device = d, list(classes), cfg, device
        self.model = Denoiser(d, len(classes), cfg.hidden, cfg.depth, cfg.dropout).to(device)
        self.ema = copy.deepcopy(self.model).eval()
        self.ab = torch.tensor(cosine_alpha_bar(cfg.T), dtype=torch.float32, device=device)

    # ------------------------------------------------------------ train
    def fit(self, Z: np.ndarray, y: np.ndarray):
        cfg = self.cfg
        X = torch.tensor(Z, dtype=torch.float32, device=self.device)
        cidx = {c: i for i, c in enumerate(self.classes)}
        yi = torch.tensor(np.array([cidx[v] for v in y]), device=self.device)
        cnt = torch.bincount(yi, minlength=len(self.classes)).float()
        w = cnt[yi].pow(-cfg.class_balance_power)
        opt = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg.steps)
        self.model.train()
        run = 0.0
        for step in range(1, cfg.steps + 1):
            idx = torch.multinomial(w, min(cfg.batch_size, len(w)), replacement=True)
            x0, yc = X[idx], yi[idx].clone()
            yc[torch.rand(len(yc), device=self.device) < cfg.p_uncond] = len(self.classes)
            t = torch.randint(0, cfg.T, (len(idx),), device=self.device)
            ab = self.ab[t][:, None]
            eps = torch.randn_like(x0)
            xt = ab.sqrt() * x0 + (1 - ab).sqrt() * eps
            loss = F.mse_loss(self.model(xt, t, yc), eps)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            opt.step(); sched.step()
            with torch.no_grad():
                for pe, pm in zip(self.ema.parameters(), self.model.parameters()):
                    pe.mul_(cfg.ema).add_(pm.detach(), alpha=1 - cfg.ema)
            run = 0.98 * run + 0.02 * loss.item() if step > 1 else loss.item()
            if step % cfg.log_every == 0 or step == cfg.steps:
                LOG.info("diffusion step %d/%d loss %.4f", step, cfg.steps, run)
        self.ema.eval()
        return self

    def state_dict(self):
        return {"ema": self.ema.state_dict(), "classes": self.classes, "d": self.d, "cfg": self.cfg.__dict__}

    @classmethod
    def from_state(cls, st, device):
        obj = cls(st["d"], st["classes"], DiffusionConfig(**st["cfg"]), device)
        obj.ema.load_state_dict(st["ema"])
        return obj

    # ----------------------------------------------------------- sample
    def _eps(self, x, t, y):
        w = self.cfg.cfg_w
        e_c = self.ema(x, t, y)
        if w == 0:
            return e_c
        e_u = self.ema(x, t, torch.full_like(y, len(self.classes)))
        return (1 + w) * e_c - w * e_u

    def sample(self, cls_label, n: int, guide: "Guidance | None" = None, batch: int = 4096,
               seed: int | None = None) -> np.ndarray:
        if seed is not None:
            torch.manual_seed(seed)
        ci = self.classes.index(cls_label)
        out = []
        for s in range(0, n, batch):
            m = min(batch, n - s)
            out.append(self._sample_batch(ci, m, guide).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, self.d), np.float32)

    def _sample_batch(self, ci, m, guide):
        cfg = self.cfg
        taus = np.linspace(cfg.T - 1, 0, cfg.sample_steps).round().astype(int)
        taus = np.unique(taus)[::-1]
        x = torch.randn(m, self.d, device=self.device)
        y = torch.full((m,), ci, dtype=torch.long, device=self.device)
        for i, t in enumerate(taus):
            ab_t = self.ab[t]
            ab_prev = self.ab[taus[i + 1]] if i + 1 < len(taus) else torch.tensor(1.0, device=self.device)
            alpha = ab_t / ab_prev
            beta = 1 - alpha
            tt = torch.full((m,), int(t), dtype=torch.long, device=self.device)
            with torch.no_grad():
                eps = self._eps(x, tt, y)
            if guide is not None and guide.active(t / cfg.T):
                # Guidance on x̂_0 (Jacobian of eps_theta dropped): x̂_0' = x̂_0 - eta_c grad V + eta_b grad B,
                # equivalent to eps~ = eps^ - sqrt(1-abar_t) g_t with g_t = sqrt(abar_t) (x̂_0' - x̂_0)/(1-abar_t).
                x0 = ((x - (1 - ab_t).sqrt() * eps) / ab_t.sqrt()).clamp(-Z_CLIP, Z_CLIP)
                x0g = guide.apply(x0, y, t / cfg.T)
                eps = (x - ab_t.sqrt() * x0g) / (1 - ab_t).sqrt()
            with torch.no_grad():
                mean = (x - beta / (1 - ab_t).sqrt() * eps) / alpha.sqrt()
                if i + 1 < len(taus):
                    var = beta * (1 - ab_prev) / (1 - ab_t)
                    x = mean + var.clamp_min(1e-20).sqrt() * torch.randn_like(x)
                else:
                    x = mean
        return x.clamp(-Z_CLIP, Z_CLIP).detach()


def _sorted(lst):
    return all(lst[i] <= lst[i + 1] for i in range(len(lst) - 1))


# ------------------------------------------------------------ guidance
@dataclass
class Guidance:
    """Dual guidance. lambda_c(t)=lambda_c(1-t/T)^gamma ; lambda_b(t)=lambda_b*1[t1 <= t/T <= t2]."""

    encoder: object
    constraints: object | None = None
    boundary: object | None = None
    lambda_c: float = 0.0
    lambda_b: float = 0.0
    gamma: float = 1.0
    t1: float = 0.2
    t2: float = 0.7
    delta: float = 1.0
    max_grad_norm: float = 1.0  # step length: every gradient is NORMALIZED to this norm (not clipped)
    inner_steps: int = 3
    stats: dict = field(default_factory=lambda: {"V": [], "B": []})

    def lam_c(self, tf):
        return self.lambda_c * (1 - tf) ** self.gamma if self.constraints is not None else 0.0

    def lam_b(self, tf):
        return self.lambda_b if (self.boundary is not None and self.t1 <= tf <= self.t2) else 0.0

    def active(self, tf):
        return self.lam_c(tf) > 0 or self.lam_b(tf) > 0

    def _clip(self, g):
        """Per-sample gradient NORMALIZATION to norm `max_grad_norm` (zero stays zero). This is not a clip:
        the scales of V and B differ widely across features, so only the direction is used and lambda_c /
        lambda_b are step lengths in the encoded (z) space. Write it this way in the paper."""
        n = g.norm(dim=1, keepdim=True)
        return torch.nan_to_num(torch.where(n > 1e-12, g / (n + 1e-12) * self.max_grad_norm, torch.zeros_like(g)))

    def apply(self, x0, y, tf):
        lc, lb = self.lam_c(tf), self.lam_b(tf)
        x = x0.detach()
        for _ in range(self.inner_steps):
            x = x.requires_grad_(True)
            step = torch.zeros_like(x)
            if lc > 0:
                num, soft = self.encoder.decode_torch(x)
                V = self.constraints.energy_torch(num, soft, self.encoder)
                (gV,) = torch.autograd.grad(V.sum(), x)
                step = step - lc * self._clip(gV)
                self.stats["V"].append(float(V.mean().detach()))
            if lb > 0:
                B = self.boundary.reward_torch(x, y, self.delta)
                (gB,) = torch.autograd.grad(B.sum(), x)
                step = step + lb * self._clip(gB)
                self.stats["B"].append(float(B.mean().detach()))
            x = (x.detach() + step).clamp(-Z_CLIP, Z_CLIP)
        return x
