"""Regression metrics, NN-vs-baselines table, monotonicity check."""
import numpy as np
from config import *
from nn.model import to_model_input
from nn.baseline import AvgMarkBaseline, RidgeBaseline
from nn.predict import NNPredictor


def metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    ss = ((y - y.mean()) ** 2).sum()
    return {"MAE": float(np.abs(y - p).mean()), "RMSE": float(np.sqrt(((y - p) ** 2).mean())),
            "R2": float(1 - ((y - p) ** 2).sum() / ss)}


def nn_vs_baselines(model, df_train, df_test):
    y = df_test[TARGET].values; X = df_test[FEATURES].values
    return {"NN (12-16-8-1)": metrics(y, model.predict(to_model_input(X))),
            "Baseline: avg previous mark": metrics(y, AvgMarkBaseline.fit(df_train).predict_raw(X)),
            "Baseline: ridge regression": metrics(y, RidgeBaseline().fit(df_train).predict_raw(X))}


def monotonicity_report(model, profiles, k_max=K_MAX, eps=1e-6):
    """Raw (un-fixed) NN predictions: does the predicted mark ever DROP when sessions go up?"""
    pred = NNPredictor(model); viol = pairs = 0; worst = 0.0; bad_series = 0; series = 0
    for p in profiles:
        for row in pred.marks_table(p, k_max, monotone=False):
            d = np.diff(row); series += 1; pairs += len(d)
            v = (d < -eps).sum(); viol += int(v); bad_series += int(v > 0)
            if v: worst = max(worst, float(-d.min()))
    return {"series": series, "series_with_violation": bad_series, "violating_steps": viol, "steps": pairs,
            "max_drop_marks": worst}
