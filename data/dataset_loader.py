"""Loading the processed table, chronological split, student profiles for the planner."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from config import *


def pres_key(p: str):
    return (int(p[:4]), 0 if p[4] == "B" else 1)


def load_processed(path) -> pd.DataFrame:
    return pd.read_csv(path)


def split_by_presentation(df: pd.DataFrame):
    """Train on all earlier presentations, test on the LAST one (chronological split)."""
    order = sorted(df.code_presentation.unique(), key=pres_key)
    return df[df.code_presentation != order[-1]].copy(), df[df.code_presentation == order[-1]].copy(), order


@dataclass
class StudentProfile:
    student_id: int
    raw: np.ndarray               # (n_assign, 12) raw feature rows of the student's assignments
    weeks_avail: list[int]        # weeks before each deadline, counted from the start of planning
    true_scores: list[float]


def make_profiles(df_test: pd.DataFrame, n_profiles: int, seed: int, n_assign: int = N_ASSIGN) -> list[StudentProfile]:
    rng = np.random.default_rng(seed)
    cand = [sid for sid, g in df_test.groupby("id_student") if len(g) >= n_assign]
    pick = rng.choice(cand, size=min(n_profiles, len(cand)), replace=False)
    out = []
    for sid in pick:
        g = df_test[df_test.id_student == sid].sort_values("date").head(n_assign)
        wd = g.week_deadline.values
        avail = np.clip(wd - (wd[0] - PLAN_HORIZON_WEEKS), 1, 16).astype(int).tolist()
        out.append(StudentProfile(int(sid), g[FEATURES].values.astype(float), avail, g.score.tolist()))
    return out
