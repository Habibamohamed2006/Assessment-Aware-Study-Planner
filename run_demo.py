"""python run_demo.py  -> input, solution, 3 demos, 30 student profiles; plots in results/, model in models/"""
import argparse, copy
from pathlib import Path
import numpy as np
from config import *
from data.dataset_loader import load_processed, split_by_presentation, make_profiles
from nn.model import MLP
from nn.train import fit_model
from nn.baseline import AvgMarkBaseline
from nn.predict import NNPredictor, AvgPredictor
from integration.planner import plan, describe
from eval.experiments import target_rule, planner_batch
from eval.metrics import monotonicity_report
from eval import plots
from csp import StudyCSP, CSPConfig

ap = argparse.ArgumentParser(); ap.add_argument("--data", default="data/processed/oulad_processed.csv")
ap.add_argument("--seed", type=int, default=SEED); ap.add_argument("--spw", type=int, default=SLOTS_PER_WEEK)
ap.add_argument("--algo", default="astar", choices=["ucs", "greedy", "astar"])
a = ap.parse_args(); R = Path("results"); R.mkdir(exist_ok=True); Path("models").mkdir(exist_ok=True)
df = load_processed(a.data); tr, te, order = split_by_presentation(df)
mp = Path("models") / f"mlp_seed{a.seed}.npz"
if mp.exists(): model = MLP.load(mp)
else: model = fit_model(tr, seed=a.seed); model.save(mp)
nn, avg = NNPredictor(model), AvgPredictor(AvgMarkBaseline.fit(tr))
profiles = make_profiles(te, 30, a.seed)
if len(profiles) < 30:
    raise RuntimeError(f"The selected course/presentation provides only {len(profiles)} eligible profiles; at least 30 are required. Choose another course or check the data.")

print("=" * 78, "\nDEMO 1 - one student, NN + Search + CSP\n" + "=" * 78)
chosen = None
for candidate in profiles:
    candidate_target = target_rule(nn.marks_table(candidate))
    candidate_result = plan(candidate, nn, candidate_target, a.spw, a.algo)
    if candidate_result.status == "OK" and candidate_result.cost and candidate_result.cost >= 3:
        chosen = (candidate, candidate_target, candidate_result)
        break
if chosen is None:
    # Still provide a useful, deterministic example if no profile meets the preferred display condition.
    p = profiles[0]
    tgt = target_rule(nn.marks_table(p))
    res = plan(p, nn, tgt, a.spw, a.algo)
else:
    p, tgt, res = chosen
print(f"Student {p.student_id}; input rows (clicks g1,g2,g3, last-week, prev_mean, prev_last, n_prev, has_prev, tma, cma, weight, week):")
print(np.round(p.raw, 1)); print("weeks before deadlines:", p.weeks_avail, "| true marks (hidden from planner):", p.true_scores)
print(describe(res)); plots.plot_cooperation(res, p, R / "demo_1_cooperation.png", f"Student {p.student_id}: NN + Search + CSP")
for al in ["ucs", "greedy", "astar"]:
    r = plan(p, nn, tgt, a.spw, al); print(f"  {al:>6}: cost={r.cost} expanded={r.search.expanded} generated={r.search.generated}")

print("\n" + "=" * 78, "\nDEMO 2 - NN on vs NN off (average-mark predictor)\n" + "=" * 78)
off = plan(p, avg, tgt, a.spw, a.algo); print("NN ON :", res.status, res.allocation); print("NN OFF:", off.message if off.status != "OK" else off.allocation)
plots.plot_cooperation(off, p, R / "demo_2_nn_off.png", "NN OFF (avg predictor ignores effort)")

print("\n" + "=" * 78, "\nDEMO 3 - NN says 'enough' but the CSP rejects the schedule\n" + "=" * 78)
found = None
for q in profiles:
    q2 = copy.deepcopy(q); q2.weeks_avail = [1, 6, 10]
    for goal in ["mean", "each"]:
        r = plan(q2, nn, target_rule(nn.marks_table(q)), 2, "ucs", goal)
        if r.search.csp_rejected > 0 and (found is None or (r.status == "OK" and found[1].status != "OK")):
            found = (q2, r)
if found:
    q2, r = found; print(describe(r))
    plots.plot_cooperation(r, q2, R / "demo_3_csp_rejects.png", "NN says enough, CSP rejects (red dots)")
else:
    print("No rejection case found for this trained model and sampled profiles. The integration test suite includes a deterministic CSP-rejection case.")

print("\n" + "=" * 78, "\nCSP PRUNING - BT vs BT+FC vs BT+FC+AC-3\n" + "=" * 78)
demo_counts = res.allocation if res.allocation is not None else res.best_effort["allocation"]
for cfg in [CSPConfig(False, False), CSPConfig(True, False), CSPConfig(True, True)]:
    csp_result = StudyCSP(demo_counts, p.weeks_avail, a.spw).solve(cfg)
    st = csp_result.stats
    print(f"{cfg.name:12} status={csp_result.status:10} assignments={st.assignments:6} backtracks={st.backtracks:6} checks={st.checks:8} FC_prunings={st.fc_prunings:6} AC3_prunings={st.ac3_prunings:6}")

print("\n" + "=" * 78, "\n30 student profiles (allocation, timetable, predicted marks) + monotonicity\n" + "=" * 78)
b = planner_batch(profiles, nn, a.spw, a.algo); b.to_csv(R / "planner_30_profiles.csv", index=False)
print(b.drop(columns=["timetable"]).head(10).to_string(index=False)); print("status counts:", b.status.value_counts().to_dict())
print("Monotonicity (raw NN):", monotonicity_report(model, profiles)); print("\nFigures in results/, full table results/planner_30_profiles.csv")
