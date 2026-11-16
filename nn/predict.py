"""What-if prediction: effort (sessions) -> predicted marks, used by the planner."""
import numpy as np
from config import *
from nn.model import MLP, to_model_input
from nn.baseline import AvgMarkBaseline


def add_sessions(raw_row: np.ndarray, k: int) -> np.ndarray:
    """What-if rule (GENERATED): k sessions add k*SESSION_CLICKS clicks, split across the 3 groups in proportion
    to the student's own mix (equal split if no clicks); SESSION_LAST_WEEK of them fall in the last week."""
    r = np.array(raw_row, dtype=float, copy=True)
    g = r[:3]
    share = g / g.sum() if g.sum() > 0 else np.ones(3) / 3
    extra = k * SESSION_CLICKS
    r[:3] += share * extra
    r[3] += SESSION_LAST_WEEK * extra
    return r


class NNPredictor:
    """marks_table(profile, k_max)[i][k] = predicted mark of assignment i after k sessions on it."""
    name = "NN"
    def __init__(self, model: MLP): self.model = model
    def marks_table(self, profile, k_max=K_MAX, monotone=True):
        n = len(profile.raw)
        rows = np.array([add_sessions(profile.raw[i], k) for i in range(n) for k in range(k_max + 1)])
        P = self.model.predict(to_model_input(rows)).reshape(n, k_max + 1)
        if monotone:                       # enforce "more effort never lowers the prediction"
            P = np.maximum.accumulate(P, axis=1)
        return P.tolist()


class AvgPredictor:
    """NN-OFF baseline: ignores effort completely (always the average of previous marks)."""
    name = "AvgMark"
    def __init__(self, base: AvgMarkBaseline): self.base = base
    def marks_table(self, profile, k_max=K_MAX, monotone=True):
        v = self.base.predict_raw(profile.raw)
        return [[float(v[i])] * (k_max + 1) for i in range(len(v))]
