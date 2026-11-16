from tests import *
from config import FEATURES
from data.dataset_loader import split_by_presentation, make_profiles

# ---- normal
def test_pipeline_columns_and_no_nan():
    df = tiny_dataset(); assert set(FEATURES + ["score"]) <= set(df.columns) and not df[FEATURES + ["score"]].isna().any().any()
def test_no_demographic_columns_and_12_features():
    assert len(FEATURES) == 12 and not {"gender", "age_band", "region", "disability", "imd_band", "highest_education"} & set(tiny_dataset().columns)
def test_exam_rows_are_excluded():
    assert set(tiny_dataset().assessment_type) <= {"TMA", "CMA"}
def test_chronological_split_test_is_last_presentation():
    tr, te, order = split_by_presentation(tiny_dataset())
    assert set(te.code_presentation) == {order[-1]} and order[-1] not in set(tr.code_presentation)
def test_previous_scores_only_use_earlier_assessments():
    df = tiny_dataset(); first = df[df.n_prev == 0]; assert (first.has_prev == 0).all() and (first.prev_mean == 50.0).all()
def test_profiles_have_three_assignments_and_deadlines():
    tr, te, _ = split_by_presentation(tiny_dataset())
    for p in make_profiles(te, 5, 0): assert p.raw.shape == (3, 12) and len(p.weeks_avail) == 3 and p.weeks_avail[0] == 4
# ---- edge
def test_edge_seeded_profiles_are_reproducible():
    _, te, _ = split_by_presentation(tiny_dataset())
    assert [p.student_id for p in make_profiles(te, 5, 3)] == [p.student_id for p in make_profiles(te, 5, 3)]
def test_edge_fewer_candidates_than_requested():
    _, te, _ = split_by_presentation(tiny_dataset()); assert len(make_profiles(te.head(9), 50, 0)) <= 50
