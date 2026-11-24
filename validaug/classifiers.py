"""NIDS classifiers (6 families). Common interface: fit(Z, y, Zva, yva) / predict_proba(Z).

By default no class weights are used, to isolate the effect of data augmentation. The optional
``class_weight`` parameter ("balanced" | "sqrt") enables cost-sensitive training; it is used only by the
non-generative control baselines (`none_cw`, `none_cw_sqrt`, see runner.py). Early stopping on
val macro-F1 (val is REAL data and never contains synthetic samples).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score


def class_weight_vector(yi, k, mode):
    """Per-class loss weights. "balanced": w_c = n / (k * n_c) (scikit-learn convention);
    "sqrt": square root of the balanced weights (damped, common for extreme imbalance).
    Weights are rescaled so that the mean weight over the training samples is 1."""
    if mode in (None, False, "none"):
        return None
    cnt = np.bincount(yi, minlength=k).astype(float)
    w = np.where(cnt > 0, len(yi) / (k * np.maximum(cnt, 1.0)), 0.0)
    if mode == "sqrt":
        w = np.sqrt(w)
    elif mode != "balanced":
        raise ValueError(f"unknown class_weight mode: {mode}")
    return w / max(float((w[yi]).mean()), 1e-12)


class _Base:
    def __init__(self, classes, seed=0, device="cpu", class_weight=None, **kw):
        self.classes = list(classes)
        self.cidx = {c: i for i, c in enumerate(self.classes)}
        self.seed, self.device, self.kw = seed, device, kw
        self.class_weight = class_weight

    def _yi(self, y):
        return np.array([self.cidx[v] for v in y])


class XGBClf(_Base):
    def fit(self, Z, y, Zva, yva):
        import xgboost as xgb

        p = dict(n_estimators=600, max_depth=8, learning_rate=0.1, subsample=0.9, colsample_bytree=0.9,
                 tree_method="hist", early_stopping_rounds=30,
                 eval_metric="mlogloss" if len(self.classes) > 2 else "logloss", n_jobs=-1,
                 random_state=self.seed, verbosity=0)
        if str(self.device).startswith("cuda"):
            p["device"] = "cuda"
        p.update(self.kw)
        if p.get("early_stopping_rounds") is None:
            p.pop("early_stopping_rounds", None)
        self.m = xgb.XGBClassifier(**p)
        okc = np.isin(np.asarray(yva, dtype=object), self.classes)
        yv = self._yi(np.asarray(yva, dtype=object)[okc])
        Zva = Zva[okc]
        ok = np.isin(yv, np.unique(self._yi(y)))
        eval_set = [(Zva[ok], yv[ok])] if "early_stopping_rounds" in p and ok.any() else None
        yi = self._yi(y)
        w = class_weight_vector(yi, len(self.classes), self.class_weight)
        self.m.fit(Z, yi, sample_weight=None if w is None else w[yi], eval_set=eval_set, verbose=False)
        return self

    def predict_proba(self, Z):
        return _pad(self.m.predict_proba(Z), self.m.classes_, len(self.classes))


class RFClf(_Base):
    def fit(self, Z, y, Zva, yva):
        from sklearn.ensemble import RandomForestClassifier

        p = dict(n_estimators=300, n_jobs=-1, random_state=self.seed, min_samples_leaf=1)
        p.update(self.kw)
        yi = self._yi(y)
        w = class_weight_vector(yi, len(self.classes), self.class_weight)
        self.m = RandomForestClassifier(**p).fit(Z, yi, sample_weight=None if w is None else w[yi])
        return self

    def predict_proba(self, Z):
        return _pad(self.m.predict_proba(Z), self.m.classes_, len(self.classes))


def _pad(P, present, k):
    out = np.zeros((len(P), k))
    out[:, np.asarray(present, int)] = P
    return out


# --------------------------------------------------------------- torch
class MLPNet(nn.Module):
    def __init__(self, d, k, h=256, p=0.2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(p),
                                 nn.Linear(h, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(p), nn.Linear(h, k))

    def forward(self, x):
        return self.net(x)


class FeatureTokenizer(nn.Module):
    """Per-feature linear embedding (as in FT-Transformer): x (n, d) -> tokens (n, d, k).
    Feeding raw scalars (one channel) to CNN/LSTM over ~100-300 encoded dimensions trains poorly
    (one-hot blocks of +-1, very long sequences); learned per-feature embeddings fix this."""

    def __init__(self, d, k):
        super().__init__()
        self.w = nn.Parameter(torch.randn(d, k) * (1.0 / k ** 0.5))
        self.b = nn.Parameter(torch.zeros(d, k))

    def forward(self, x):
        return x[:, :, None] * self.w[None] + self.b[None]


class CNN1DNet(nn.Module):
    """1D-CNN over the sequence of embedded features."""

    def __init__(self, d, k, c=64, d_token=16):
        super().__init__()
        self.tok = FeatureTokenizer(d, d_token)
        self.conv = nn.Sequential(nn.Conv1d(d_token, c, 3, padding=1), nn.BatchNorm1d(c), nn.ReLU(),
                                  nn.Conv1d(c, c, 3, padding=1), nn.BatchNorm1d(c), nn.ReLU(),
                                  nn.AdaptiveMaxPool1d(8))
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(c * 8, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, k))

    def forward(self, x):
        return self.fc(self.conv(self.tok(x).transpose(1, 2)))


class LSTMNet(nn.Module):
    """Bidirectional LSTM over the sequence of embedded features (a common NIDS setting, with
    per-feature embeddings instead of raw scalars), mean + max pooling."""

    def __init__(self, d, k, h=32, d_token=16):
        super().__init__()
        self.tok = FeatureTokenizer(d, d_token)
        self.lstm = nn.LSTM(d_token, h, batch_first=True, bidirectional=True)
        self.norm = nn.LayerNorm(4 * h)
        self.fc = nn.Sequential(nn.Linear(4 * h, 32), nn.ReLU(), nn.Linear(32, k))

    def forward(self, x):
        o, _ = self.lstm(self.tok(x))
        return self.fc(self.norm(torch.cat([o.mean(1), o.amax(1)], 1)))


class FTTransformerNet(nn.Module):
    """Compact FT-Transformer (Gorishniy et al., 2021): every encoded dimension is one token."""

    def __init__(self, d, k, dt=32, layers=2, heads=4):
        super().__init__()
        self.w = nn.Parameter(torch.randn(d, dt) * 0.02)
        self.b = nn.Parameter(torch.zeros(d, dt))
        self.cls = nn.Parameter(torch.zeros(1, 1, dt))
        enc = nn.TransformerEncoderLayer(dt, heads, dt * 4, dropout=0.1, batch_first=True, norm_first=True)
        self.tr = nn.TransformerEncoder(enc, layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(dt), nn.ReLU(), nn.Linear(dt, k))

    def forward(self, x):
        tok = x[:, :, None] * self.w[None] + self.b[None]
        tok = torch.cat([self.cls.expand(len(x), -1, -1), tok], 1)
        return self.head(self.tr(tok)[:, 0])


NETS = {"mlp": MLPNet, "cnn1d": CNN1DNet, "lstm": LSTMNet, "ft_transformer": FTTransformerNet}


class TorchClf(_Base):
    def __init__(self, classes, arch="mlp", seed=0, device="cpu", epochs=30, batch_size=512, lr=1e-3,
                 patience=5, max_val=50000, min_epochs=5, class_weight=None, **kw):
        super().__init__(classes, seed, device, class_weight=class_weight)
        self.arch, self.epochs, self.bs, self.lr, self.patience, self.max_val = arch, epochs, batch_size, lr, patience, max_val
        self.min_epochs = min_epochs
        self.net_kw = kw

    def fit(self, Z, y, Zva, yva):
        torch.manual_seed(self.seed)
        dev = self.device
        X = torch.tensor(Z, dtype=torch.float32)
        Y = torch.tensor(self._yi(y))
        rng = np.random.default_rng(self.seed)
        ok = np.isin(np.asarray(yva, dtype=object), self.classes)  # val may hold classes absent from train
        Zva, yva = Zva[ok], np.asarray(yva, dtype=object)[ok]
        if len(Zva) > self.max_val:
            ix = rng.choice(len(Zva), self.max_val, replace=False); Zva, yva = Zva[ix], yva[ix]
        yv = self._yi(yva)
        self.net = NETS[self.arch](Z.shape[1], len(self.classes), **self.net_kw).to(dev)
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, self.epochs)
        cw = class_weight_vector(Y.numpy(), len(self.classes), self.class_weight)
        cw = None if cw is None else torch.tensor(cw, dtype=torch.float32, device=dev)
        best, bad, best_state = -1, 0, None
        for ep in range(self.epochs):
            self.net.train()
            perm = torch.randperm(len(X))
            for s in range(0, len(X), self.bs):
                b = perm[s:s + self.bs]
                if len(b) < 2:
                    continue
                loss = F.cross_entropy(self.net(X[b].to(dev)), Y[b].to(dev), weight=cw)
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step()
            sched.step()
            f1 = f1_score(yv, self.predict_proba(Zva).argmax(1), average="macro", zero_division=0)
            if f1 > best + 1e-4:
                best, bad = f1, 0
                best_state = {k: v.detach().clone() for k, v in self.net.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience and ep + 1 >= self.min_epochs:
                    break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        self.best_val_f1, self.epochs_run = best, ep + 1
        return self

    @torch.no_grad()
    def predict_proba(self, Z):
        self.net.eval()
        out = []
        for s in range(0, len(Z), 8192):
            out.append(torch.softmax(self.net(torch.tensor(Z[s:s + 8192], dtype=torch.float32, device=self.device)), 1).cpu())
        return torch.cat(out).numpy()


def make_classifier(name, classes, seed, device, params=None):
    params = dict(params or {})
    if name == "xgboost":
        return XGBClf(classes, seed, device, **params)
    if name == "rf":
        return RFClf(classes, seed, device, **params)
    if name in NETS:
        return TorchClf(classes, arch=name, seed=seed, device=device, **params)
    raise KeyError(name)
