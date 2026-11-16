from tests import *
from config import FEATURES, SESSION_CLICKS
from data.dataset_loader import split_by_presentation, make_profiles
from nn.model import to_model_input
from nn.train import fit_model
from nn.baseline import AvgMarkBaseline, RidgeBaseline
from nn.predict import NNPredictor, AvgPredictor, add_sessions
from eval.metrics import metrics, monotonicity_report

def _fit():
    tr, te, _ = split_by_presentation(tiny_dataset()); return tr, te, fit_model(tr, seed=0)
# ---- normal
def test_nn_beats_average_baseline():
    tr, te, m = _fit(); y = te.score.values
    assert metrics(y, m.predict(to_model_input(te[FEATURES].values)))["RMSE"] < metrics(y, AvgMarkBaseline.fit(tr).predict_raw(te[FEATURES].values))["RMSE"]
def test_nn_beats_ridge_or_is_close():
    tr, te, m = _fit(); y = te.score.values; X = te[FEATURES].values
    assert metrics(y, m.predict(to_model_input(X)))["RMSE"] < metrics(y, RidgeBaseline().fit(tr).predict_raw(X))["RMSE"] + 0.5
def test_predictions_within_0_100():
    _, te, m = _fit(); p = m.predict(to_model_input(te[FEATURES].values)); assert p.min() >= 0 and p.max() <= 100
def test_training_loss_decreases():
    _, _, m = _fit(); assert m.history[-1][0] < m.history[0][0]
def test_add_sessions_adds_expected_clicks():
    r = np.array([10, 10, 20, 5, 50, 50, 0, 0, 1, 0, 10, 3.0]); out = add_sessions(r, 2)
    assert abs(out[:3].sum() - r[:3].sum() - 2 * SESSION_CLICKS) < 1e-9 and out[4:].tolist() == r[4:].tolist()
def test_save_and_load_roundtrip(tmp_path=None):
    import tempfile, os; from nn.model import MLP
    _, te, m = _fit(); f = os.path.join(tempfile.mkdtemp(), "m.npz"); m.save(f); X = to_model_input(te[FEATURES].values[:20])
    assert np.allclose(MLP.load(f).predict(X), m.predict(X))
# ---- edge
def test_edge_monotone_wrapper_never_decreases():
    _, te, m = _fit()
    for p in make_profiles(te, 10, 0):
        for row in NNPredictor(m).marks_table(p): assert all(b >= a - 1e-9 for a, b in zip(row, row[1:]))
def test_edge_monotonicity_report_has_fields():
    _, te, m = _fit(); r = monotonicity_report(m, make_profiles(te, 5, 0)); assert r["series"] == 15 and r["violating_steps"] >= 0
def test_edge_zero_clicks_student_gets_equal_split():
    out = add_sessions(np.zeros(12), 3); assert np.allclose(out[:3], SESSION_CLICKS * 3 / 3)
# ---- impossible-ish: the average predictor cannot respond to effort
def test_avg_predictor_ignores_effort():
    class Pr: raw = np.array([[0, 0, 0, 0, 60, 60, 1, 1, 1, 0, 10, 3.0]])
    row = AvgPredictor(AvgMarkBaseline(50.0)).marks_table(Pr)[0]; assert len(set(row)) == 1 and row[0] == 60
