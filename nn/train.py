"""Train the NN on earlier presentations and evaluate it on the latest one.
Run: python -m nn.train --data data/processed/oulad_processed.csv --seed 42
"""
import argparse
from pathlib import Path
import numpy as np
from config import *
from data.dataset_loader import load_processed, split_by_presentation
from nn.model import MLP, to_model_input
from nn.baseline import AvgMarkBaseline, RidgeBaseline
from eval.metrics import metrics


def fit_model(df_train, seed=0, **kw) -> MLP:
    rng = np.random.default_rng(seed)
    X, y = to_model_input(df_train[FEATURES].values), df_train[TARGET].values.astype(float)
    if len(X) < 10:
        raise ValueError(f"Need at least 10 training rows; received {len(X)}")
    idx = rng.permutation(len(X)); nv = max(1, int(0.15 * len(X)))
    va, tr = idx[:nv], idx[nv:]
    return MLP(seed=seed).fit(X[tr], y[tr], X[va], y[va], seed=seed, **kw)


def main():
    ap = argparse.ArgumentParser(description="Train the study planner NN and compare baselines.")
    ap.add_argument("--data", default="data/processed/oulad_processed.csv")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out", default="models/study_planner_mlp.npz")
    a = ap.parse_args()
    df = load_processed(a.data)
    train, test, order = split_by_presentation(df)
    if len(train) == 0 or len(test) == 0:
        raise ValueError("Chronological split requires at least two presentations with rows.")
    model = fit_model(train, seed=a.seed)
    X = to_model_input(test[FEATURES].values); y = test[TARGET].values
    results = {"NN": metrics(y, model.predict(X)),
               "Average-mark baseline": metrics(y, AvgMarkBaseline.fit(train).predict_raw(test[FEATURES].values)),
               "Ridge baseline": metrics(y, RidgeBaseline().fit(train).predict_raw(test[FEATURES].values))}
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True); model.save(out)
    print(f"Train presentations: {order[:-1]} | held-out presentation: {order[-1]}")
    print(f"Saved trained model: {out}")
    for name, values in results.items():
        print(name, "|", " | ".join(f"{k}={v:.4f}" for k, v in values.items()))


if __name__ == "__main__":
    main()
