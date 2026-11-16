"""Baselines the NN must beat: average previous mark, ridge regression."""
import numpy as np
from config import *
from nn.model import to_model_input


class AvgMarkBaseline:
    """Predict the mean of the student's previous marks (global train mean if none)."""
    def __init__(self, train_mean: float): self.train_mean = train_mean
    @classmethod
    def fit(cls, df_train): return cls(float(df_train[TARGET].mean()))
    def predict_raw(self, raw):
        raw = np.atleast_2d(raw); i_pm, i_hp = FEATURES.index("prev_mean"), FEATURES.index("has_prev")
        return np.where(raw[:, i_hp] > 0, raw[:, i_pm], self.train_mean)


class RidgeBaseline:
    def __init__(self, lam=1.0): self.lam = lam
    def fit(self, df_train):
        X = to_model_input(df_train[FEATURES].values); self.mu, self.sd = X.mean(0), X.std(0) + 1e-8
        Xs = np.c_[np.ones(len(X)), (X - self.mu) / self.sd]; y = df_train[TARGET].values
        A = Xs.T @ Xs + self.lam * np.eye(Xs.shape[1]); A[0, 0] -= self.lam
        self.w = np.linalg.solve(A, Xs.T @ y); return self
    def predict_raw(self, raw):
        X = to_model_input(np.atleast_2d(raw)); return np.clip(np.c_[np.ones(len(X)), (X - self.mu) / self.sd] @ self.w, 0, 100)
