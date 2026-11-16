"""Shared test helpers (run tests from the project root: python -m pytest -q)."""
import sys, tempfile, functools
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np


class FakePredictor:
    """Linear toy predictor: mark_i(k) = base_i + gain_i*k (known answers for search/integration tests)."""
    name = "fake"
    def __init__(self, base, gain): self.base, self.gain = base, gain
    def marks_table(self, profile, k_max=8, monotone=True):
        return [[min(100.0, b + g * k) for k in range(k_max + 1)] for b, g in zip(self.base, self.gain)]


class P:   # minimal student profile
    def __init__(self, weeks): self.weeks_avail = weeks; self.raw = np.zeros((len(weeks), 12)); self.student_id = 0


@functools.lru_cache(maxsize=1)
def tiny_dataset():
    """Small synthetic OULAD-schema dataset run through the REAL preprocessing pipeline (for tests only)."""
    from data.make_data import make_synthetic_raw
    from data.preprocessing import build_processed
    d = Path(tempfile.mkdtemp()); make_synthetic_raw(d / "raw", seed=1, n_students=250)
    return build_processed(d / "raw", "SYN", d / "p.csv")
