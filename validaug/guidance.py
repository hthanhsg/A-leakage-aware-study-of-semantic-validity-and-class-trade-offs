"""Ensemble of auxiliary classifiers f_phi for boundary guidance (paper Sec. 4).

m_y(x) = f(x)_y - max_{j!=y} f(x)_j ;  B(x,y) = -(m_y(x) - delta)^2, averaged over the E members.
Trained on real training data only. Small Gaussian noise is added during training so that gradients
on x_0-hat (still noisy at intermediate steps) are more stable.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import LOG


def _mlp(d, k, h=256):
    return nn.Sequential(nn.Linear(d, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, k))


class BoundaryEnsemble:
    def __init__(self, d, classes, n_members=3, hidden=256, epochs=20, lr=1e-3,
                 noise_std=0.1, batch_size=1024, device="cpu", seed=0):
        self.classes = list(classes)
        self.cidx = {c: i for i, c in enumerate(self.classes)}
        self.device = device
        self.cfg = dict(n_members=n_members, hidden=hidden, epochs=epochs, lr=lr,
                        noise_std=noise_std, batch_size=batch_size, seed=seed)
        g = torch.Generator().manual_seed(seed)
        self.members = []
        for _ in range(n_members):
            torch.manual_seed(int(torch.randint(0, 1 << 30, (1,), generator=g)))
            self.members.append(_mlp(d, len(self.classes), hidden).to(device))

    def fit(self, Z, y):
        c = self.cfg
        X = torch.tensor(Z, dtype=torch.float32, device=self.device)
        Y = torch.tensor([self.cidx[v] for v in y], device=self.device)
        cnt = torch.bincount(Y, minlength=len(self.classes)).float().clamp_min(1)
        wcls = (cnt.sum() / cnt) ** 0.5
        wcls = wcls / wcls.mean()
        rng = np.random.default_rng(c["seed"])
        for k, net in enumerate(self.members):
            boot = torch.tensor(rng.integers(0, len(X), len(X)), device=self.device)  # bootstrap
            opt = torch.optim.AdamW(net.parameters(), lr=c["lr"], weight_decay=1e-4)
            net.train()
            for _ in range(c["epochs"]):
                perm = boot[torch.randperm(len(boot), device=self.device)]
                for s in range(0, len(perm), c["batch_size"]):
                    b = perm[s:s + c["batch_size"]]
                    xb = X[b] + c["noise_std"] * torch.randn_like(X[b])
                    loss = F.cross_entropy(net(xb), Y[b], weight=wcls)
                    opt.zero_grad(); loss.backward(); opt.step()
            net.eval()
        LOG.info("Boundary ensemble: %d members trained", len(self.members))
        return self

    def _margins_torch(self, x, yi):
        out = []
        for net in self.members:
            lg = net(x)
            ly = lg.gather(1, yi[:, None]).squeeze(1)
            other = lg.masked_fill(F.one_hot(yi, lg.shape[1]).bool(), -1e9).max(1).values
            out.append(ly - other)
        return torch.stack(out, 0)  # (E, n)

    def reward_torch(self, x0, yi, delta):
        """B(x,y) = -mean_e (m_y^e(x) - delta)^2 ; yi are class indices in the diffusion class order."""
        return -((self._margins_torch(x0, yi) - delta) ** 2).mean(0)

    @torch.no_grad()
    def margin(self, Z, y) -> np.ndarray:
        X = torch.tensor(Z, dtype=torch.float32, device=self.device)
        Y = torch.tensor([self.cidx[v] for v in y], device=self.device)
        out = []
        for s in range(0, len(X), 8192):
            out.append(self._margins_torch(X[s:s + 8192], Y[s:s + 8192]).mean(0).cpu())
        return torch.cat(out).numpy() if out else np.zeros(0)
