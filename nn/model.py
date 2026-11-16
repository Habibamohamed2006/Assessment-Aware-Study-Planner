"""Small numpy MLP (12-16-8-1, tanh, Adam, early stopping) and input transform."""
import numpy as np
from config import *


def to_model_input(raw: np.ndarray) -> np.ndarray:
    X = np.array(raw, dtype=float, copy=True)
    X[:, :CLICK_COLS] = np.log1p(X[:, :CLICK_COLS])
    return X


class MLP:
    def __init__(self, n_in=len(FEATURES), hidden=(16, 8), seed=0):
        rng = np.random.default_rng(seed)
        sizes = [n_in, *hidden, 1]
        self.W = [rng.normal(0, np.sqrt(1.0 / sizes[i]), (sizes[i], sizes[i + 1])) for i in range(len(sizes) - 1)]
        self.b = [np.zeros(sizes[i + 1]) for i in range(len(sizes) - 1)]
        self.mu = np.zeros(n_in); self.sd = np.ones(n_in)

    def _forward(self, X):
        acts = [X]
        for l, (W, b) in enumerate(zip(self.W, self.b)):
            z = acts[-1] @ W + b
            acts.append(np.tanh(z) if l < len(self.W) - 1 else z)
        return acts

    def fit(self, X, y, Xv, yv, epochs=300, lr=3e-3, batch=128, l2=1e-4, patience=25, seed=0):
        rng = np.random.default_rng(seed)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-8
        Xs, Xvs = (X - self.mu) / self.sd, (Xv - self.mu) / self.sd
        y, yv = y.reshape(-1, 1) / 100.0, yv.reshape(-1, 1) / 100.0
        params = self.W + self.b
        m = [np.zeros_like(p) for p in params]; v = [np.zeros_like(p) for p in params]
        best, best_p, wait, t = np.inf, None, 0, 0
        self.history = []
        for ep in range(epochs):
            idx = rng.permutation(len(Xs))
            for s in range(0, len(idx), batch):
                bi = idx[s:s + batch]
                acts = self._forward(Xs[bi])
                delta = 2 * (acts[-1] - y[bi]) / len(bi)
                gW, gb = [None] * len(self.W), [None] * len(self.W)
                for l in reversed(range(len(self.W))):
                    gW[l] = acts[l].T @ delta + l2 * self.W[l]
                    gb[l] = delta.sum(0)
                    if l > 0:
                        delta = (delta @ self.W[l].T) * (1 - acts[l] ** 2)
                grads = gW + gb; t += 1
                for i, (p, g) in enumerate(zip(params, grads)):
                    m[i] = 0.9 * m[i] + 0.1 * g; v[i] = 0.999 * v[i] + 0.001 * g * g
                    p -= lr * (m[i] / (1 - 0.9 ** t)) / (np.sqrt(v[i] / (1 - 0.999 ** t)) + 1e-8)
            tr = float(np.mean((self._forward(Xs)[-1] - y) ** 2)); va = float(np.mean((self._forward(Xvs)[-1] - yv) ** 2))
            self.history.append((tr, va))
            if va < best - 1e-6:
                best, best_p, wait = va, [p.copy() for p in params], 0
            else:
                wait += 1
                if wait >= patience:
                    break
        for p, q in zip(params, best_p):
            p[...] = q
        return self

    def predict(self, X):
        out = self._forward((X - self.mu) / self.sd)[-1].ravel() * 100.0
        return np.clip(out, 0, 100)

    def save(self, path):
        np.savez(path, mu=self.mu, sd=self.sd, **{f"W{i}": w for i, w in enumerate(self.W)},
                 **{f"b{i}": b for i, b in enumerate(self.b)})

    @classmethod
    def load(cls, path):
        z = np.load(path); n = len([k for k in z.files if k.startswith("W")])
        m = cls(); m.W = [z[f"W{i}"] for i in range(n)]; m.b = [z[f"b{i}"] for i in range(n)]
        m.mu, m.sd = z["mu"], z["sd"]
        return m
