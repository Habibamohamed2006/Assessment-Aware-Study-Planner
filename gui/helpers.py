"""GUI helper layer: the ONLY place where the GUI touches the project's core modules.

No Streamlit / Plotly import here, so everything in this file is unit-testable.
Nothing in this module re-implements NN, Search or CSP logic: it loads the saved model, calls
`integration.planner.plan`, `csp.StudyCSP`, `eval.metrics`, ... and reshapes their outputs for display.
"""
from __future__ import annotations

import copy
import dataclasses
import functools
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:                      # make `import config`, `import nn` ... work from any cwd
    sys.path.insert(0, str(ROOT))

from config import FEATURES, K_MAX, SEED, SLOTS_PER_WEEK, TARGET                       # noqa: E402
from csp import CSPConfig, StudyCSP, capacity_ok, check_solution                       # noqa: E402
from data.dataset_loader import StudentProfile, load_processed, make_profiles, split_by_presentation  # noqa: E402
from eval.experiments import target_rule                                                # noqa: E402
from eval.metrics import monotonicity_report, nn_vs_baselines                           # noqa: E402
from integration.planner import PlanResult, plan                                        # noqa: E402
from nn.baseline import AvgMarkBaseline                                                 # noqa: E402
from nn.model import MLP, to_model_input                                                # noqa: E402
from nn.predict import AvgPredictor, NNPredictor                                        # noqa: E402
from nn.train import fit_model                                                          # noqa: E402
from search import ALGOS, build_problem                                                 # noqa: E402

DATA_PATH = ROOT / "data" / "processed" / "oulad_processed.csv"
MODEL_CANDIDATES = [ROOT / "models" / "study_planner_mlp.npz", ROOT / "models" / f"mlp_seed{SEED}.npz"]
RESULTS_DIR = ROOT / "results"

ALGO_LABELS = {"ucs": "Uniform Cost Search (UCS)", "greedy": "Greedy Best-First", "astar": "A*"}
GOAL_LABELS = {"each": "Every assignment reaches the target", "mean": "Average mark reaches the target"}
GUI_CSP_MAX_STEPS = 20_000       # assignment cut-off for the GUI (same value the project's Experiment 2 uses)
CSP_MODES = {                    # label -> (use_fc, use_ac3)
    "Backtracking only": (False, False),
    "Backtracking + Forward Checking": (True, False),
    "Backtracking + Forward Checking + AC-3": (True, True),
}
MAX_WEEKS, MAX_SPW = 16, 8       # planner clips horizons to 1-16 weeks (data/dataset_loader.py)


class GuiError(Exception):
    """Raised with a user-facing message (the pages show it with st.error)."""


# --------------------------------------------------------------------------------------- context
@dataclass
class Context:
    df: pd.DataFrame
    train: pd.DataFrame
    test: pd.DataFrame
    order: list
    model: MLP
    model_path: Path
    nn: NNPredictor
    avg: AvgPredictor
    profiles: list            # all eligible held-out student profiles (sorted by student id)
    seed: int = SEED

    def predictor(self, nn_on: bool):
        return self.nn if nn_on else self.avg

    def profile(self, student_id: int) -> StudentProfile:
        for p in self.profiles:
            if p.student_id == student_id:
                return p
        raise GuiError(f"Unknown student id {student_id}: it is not an eligible held-out profile.")


@functools.lru_cache(maxsize=2)
def load_context(seed: int = SEED, data_path: str | None = None, model_path: str | None = None) -> Context:
    """Load processed data + the SAVED model (never retrains). Cached for the life of the process."""
    dpath = Path(data_path) if data_path else DATA_PATH
    if not dpath.exists():
        raise GuiError(f"Processed data not found: {dpath}. Run `python -m data.make_data --download --module FFF` "
                       f"(or `--synthetic` for an offline smoke test) from the project root.")
    df = load_processed(dpath)
    missing = [c for c in FEATURES + [TARGET, "code_presentation", "id_student"] if c not in df.columns]
    if missing:
        raise GuiError(f"{dpath.name} is missing required columns: {missing}")
    train, test, order = split_by_presentation(df)
    if len(train) == 0 or len(test) == 0:
        raise GuiError("The chronological split needs at least two presentations with rows.")
    cands = [Path(model_path)] if model_path else MODEL_CANDIDATES
    mp = next((c for c in cands if c.exists()), None)
    if mp is None:
        raise GuiError("No trained model found in models/. Train once with "
                       "`python -m nn.train --data data/processed/oulad_processed.csv --seed 42 "
                       "--out models/study_planner_mlp.npz` (the GUI never retrains automatically).")
    model = MLP.load(mp)
    profiles = sorted(make_profiles(test, 10 ** 6, seed), key=lambda p: p.student_id)   # every eligible student
    if not profiles:
        raise GuiError("No eligible held-out student has three scored, dated assessments.")
    return Context(df, train, test, order, model, mp, NNPredictor(model),
                   AvgPredictor(AvgMarkBaseline.fit(train)), profiles, seed)


_DEFAULTS: dict = {}     # id(context) -> default student id


def pick_default_student(ctx: Context, spw: int = SLOTS_PER_WEEK, algo: str = "astar") -> int:
    """Same rule run_demo.py uses for DEMO 1: first (seeded-random order) profile that needs >= 3 sessions."""
    cache = _DEFAULTS.get(id(ctx))
    if cache is not None:
        return cache
    shuffled = make_profiles(ctx.test, 30, ctx.seed)          # exactly the 30 profiles run_demo.py uses
    chosen = ctx.profiles[0].student_id
    for p in shuffled:
        r = plan(p, ctx.nn, target_rule(ctx.nn.marks_table(p)), spw, algo)
        if r.status == "OK" and r.cost and r.cost >= 3:
            chosen = p.student_id
            break
    _DEFAULTS[id(ctx)] = chosen
    return chosen



# --------------------------------------------------------------------------------------- scenario
@dataclass(frozen=True)
class Scenario:
    student_id: int
    target: float
    spw: int
    goal: str
    weeks: tuple
    nn_on: bool


def reference_target(ctx: Context, profile: StudentProfile) -> float:
    """The project's target rule (eval.experiments.target_rule), always computed from the NN table
    (this is what run_demo.py and Experiment 3 do, so NN-ON and NN-OFF chase the same target)."""
    return target_rule(ctx.nn.marks_table(profile))


def validate_scenario(ctx: Context, sc: Scenario) -> None:
    errs = []
    try:
        profile = ctx.profile(sc.student_id)
    except GuiError as e:
        raise GuiError(str(e)) from None
    if not isinstance(sc.target, (int, float)) or isinstance(sc.target, bool) or not math.isfinite(sc.target):
        errs.append("Target mark must be a number.")
    elif not 0 <= sc.target <= 100:
        errs.append(f"Target mark must be between 0 and 100 (got {sc.target}).")
    if not isinstance(sc.spw, (int, np.integer)) or not 1 <= sc.spw <= MAX_SPW:
        errs.append(f"Study slots per week must be an integer from 1 to {MAX_SPW} (got {sc.spw}).")
    if sc.goal not in GOAL_LABELS:
        errs.append(f"Unknown goal '{sc.goal}'.")
    if len(sc.weeks) != len(profile.raw):
        errs.append(f"Expected {len(profile.raw)} deadlines, got {len(sc.weeks)}.")
    elif any((not isinstance(w, (int, np.integer))) or not 1 <= w <= MAX_WEEKS for w in sc.weeks):
        errs.append(f"Each deadline must be an integer number of weeks from 1 to {MAX_WEEKS}.")
    if errs:
        raise GuiError(" ".join(errs))


def resolve_scenario(ctx: Context, state: Mapping, nn_on: bool = True) -> Scenario:
    """Build the current scenario from Streamlit's session_state (or any mapping), with sane defaults."""
    sid = int(state.get("sel_student") or pick_default_student(ctx))
    profile = ctx.profile(sid)
    target = state.get(f"target_{sid}")
    if target is None:
        target = reference_target(ctx, profile)
    weeks = tuple(int(state.get(f"weeks_{sid}_{i}", w)) for i, w in enumerate(profile.weeks_avail))
    return Scenario(sid, float(target), int(state.get("sel_spw", SLOTS_PER_WEEK)),
                    state.get("sel_goal", "each"), weeks, bool(nn_on))


def assignment_info(ctx: Context, sc: Scenario) -> pd.DataFrame:
    p = ctx.profile(sc.student_id)
    f = {name: i for i, name in enumerate(FEATURES)}
    rows = []
    for i, r in enumerate(p.raw):
        rows.append({
            "Assignment": f"A{i + 1}",
            "Type": "TMA" if r[f["is_tma"]] > 0 else "CMA",
            "Weight (%)": float(r[f["weight"]]),
            "Real deadline (course week)": int(r[f["weeks_since_start"]]),
            "Weeks available (planner)": sc.weeks[i],
            "Clicks (4 wk before deadline)": int(r[f["clicks_g1"]] + r[f["clicks_g2"]] + r[f["clicks_g3"]]),
            "Clicks last week": int(r[f["clicks_last_week"]]),
            "Previous mean mark": float(r[f["prev_mean"]]) if r[f["has_prev"]] > 0 else None,
            "Actual mark (held out; hidden from planner)": float(p.true_scores[i]),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------- planning
def csp_config(mode: str, max_steps: int = GUI_CSP_MAX_STEPS) -> CSPConfig:
    if mode not in CSP_MODES:
        raise GuiError(f"Unknown CSP mode '{mode}'.")
    fc, ac = CSP_MODES[mode]
    return CSPConfig(fc, ac, max_steps)


@dataclass
class PlanOutcome:
    scenario: Scenario
    algo: str
    csp_mode: str
    profile: StudentProfile      # profile with the scenario's deadlines applied
    result: PlanResult
    predictor_name: str

    @property
    def ok(self) -> bool:
        return self.result.status == "OK"

    @property
    def shown(self) -> dict:
        """allocation / marks / schedule to display: the plan, or the planner's best-effort fallback."""
        r = self.result
        if r.status == "OK":
            return {"allocation": r.allocation, "marks": r.marks, "schedule": r.schedule, "best_effort": False}
        be = r.best_effort
        return {"allocation": be["allocation"], "marks": be["marks"], "schedule": be["schedule"], "best_effort": True}


def run_plan(ctx: Context, sc: Scenario, algo: str = "astar", csp_mode: str = "Backtracking + Forward Checking + AC-3") -> PlanOutcome:
    validate_scenario(ctx, sc)
    if algo not in ALGOS:
        raise GuiError(f"Unknown search algorithm '{algo}'.")
    profile = dataclasses.replace(ctx.profile(sc.student_id), weeks_avail=list(sc.weeks))
    predictor = ctx.predictor(sc.nn_on)
    res = plan(profile, predictor, sc.target, sc.spw, algo, sc.goal, csp_config(csp_mode))
    return PlanOutcome(sc, algo, csp_mode, profile, res, predictor.name)


def per_assignment_table(out: PlanOutcome) -> pd.DataFrame:
    r, s = out.result, out.shown
    rows = []
    for i, k in enumerate(s["allocation"]):
        base, mark = r.table[i][0], s["marks"][i]
        rows.append({"Assignment": f"A{i + 1}", "Weeks available": out.scenario.weeks[i], "Sessions": int(k),
                     "Predicted (no extra effort)": round(base, 1), "Predicted (with plan)": round(mark, 1),
                     "Target": out.scenario.target,
                     "Reaches target": bool(mark >= out.scenario.target - 1e-9)})
    return pd.DataFrame(rows)


@dataclass
class TimetableGrid:
    ids: np.ndarray            # (weeks, spw) 0 = free, k = assignment k (1-based)
    labels: list               # text per cell
    week_labels: list
    slot_labels: list


def timetable_grid(schedule, weeks_avail, spw) -> TimetableGrid:
    horizon = max(weeks_avail)
    ids = np.zeros((horizon, spw), dtype=int)
    labels = [[""] * spw for _ in range(horizon)]
    for a, j, w, s in schedule:
        ids[w - 1, s - 1] = a + 1
        labels[w - 1][s - 1] = f"A{a + 1}·S{j}"
    wl = []
    for w in range(1, horizon + 1):
        due = [f"A{i + 1}" for i, x in enumerate(weeks_avail) if x == w]
        wl.append(f"Week {w}" + (f"  (deadline {', '.join(due)})" if due else ""))
    return TimetableGrid(ids, labels, wl, [f"Slot {s}" for s in range(1, spw + 1)])


def timetable_frame(schedule, weeks_avail, spw) -> pd.DataFrame:
    g = timetable_grid(schedule, weeks_avail, spw)
    return pd.DataFrame([[c or "—" for c in row] for row in g.labels], index=g.week_labels, columns=g.slot_labels)


def whatif_frame(table) -> pd.DataFrame:
    """Long frame of the predictor's marks table: assignment, sessions, predicted mark."""
    return pd.DataFrame([{"Assignment": f"A{i + 1}", "Sessions": k, "Predicted mark": v}
                         for i, row in enumerate(table) for k, v in enumerate(row)])


# --------------------------------------------------------------------------------------- search
def compare_algorithms(ctx: Context, sc: Scenario, csp_mode: str = "Backtracking + Forward Checking + AC-3") -> dict:
    """Run UCS, Greedy and A* on the SAME scenario via the real planner. Returns {algo: PlanOutcome}."""
    return {a: run_plan(ctx, sc, a, csp_mode) for a in ALGOS}


def search_summary(outcomes: Mapping[str, PlanOutcome]) -> pd.DataFrame:
    rows = []
    for a, o in outcomes.items():
        r, st = o.result, o.result.search
        rows.append({"Algorithm": ALGO_LABELS[a], "key": a, "Status": r.status, "Solution cost (sessions)": r.cost,
                     "Allocation": list(r.allocation) if r.allocation else None,
                     "Expanded": st.expanded, "Generated": st.generated, "Goal tests": st.goal_tests,
                     "CSP rejected": st.csp_rejected, "CSP calls": r.csp_calls})
    return pd.DataFrame(rows)


def trace_frame(res: PlanResult) -> pd.DataFrame:
    """Expansion trace of a search run (res.trace) plus NN marks looked up in res.table."""
    _, nn_ok, _, _ = build_problem(res.table, res.target, res.goal)
    n = len(res.table)
    rows, final = [], tuple(res.allocation) if res.allocation is not None else None
    for t in res.trace:
        s = t["state"]
        marks = [res.table[i][s[i]] for i in range(n)]
        if t["rejected"]:
            status = "NN says enough → CSP rejected"
        elif final is not None and s == final and nn_ok(s):
            status = "Goal (NN ok + CSP ok)"
        else:
            status = "Expanded"
        rows.append({"Step": t["order"], "State (sessions per assignment)": str(s), **{f"A{i + 1}": s[i] for i in range(n)},
                     "g (sessions so far)": t["g"], "Priority f": round(float(t["f"]), 3),
                     "Predicted marks": [round(m, 1) for m in marks], "Min mark": round(min(marks), 1),
                     "Mean mark": round(sum(marks) / n, 1), "NN target met": bool(nn_ok(s)), "Outcome": status})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------- CSP
def capacity_report(counts, weeks, spw) -> pd.DataFrame:
    """Explains the exact feasibility test in csp.constraints.capacity_ok: by every deadline, sessions due <= slots."""
    rows, total = [], 0
    for w, i in sorted(zip(weeks, range(len(counts)))):
        total += counts[i]
        rows.append({"Deadline (weeks)": w, "Assignment": f"A{i + 1}", "Sessions due by then (cumulative)": total,
                     "Slots available by then": w * spw, "Enough slots": total <= w * spw})
    return pd.DataFrame(rows)


def run_csp_modes(counts, weeks, spw, max_steps: int = GUI_CSP_MAX_STEPS) -> dict:
    """Solve one scheduling instance with every CSP mode. Returns {mode: dict(result, timetable, valid, csp)}."""
    counts, weeks = [int(c) for c in counts], [int(w) for w in weeks]
    if len(counts) != len(weeks) or any(c < 0 for c in counts) or any(w < 1 for w in weeks) or spw < 1:
        raise GuiError("Invalid CSP instance: need equal-length counts/deadlines, counts >= 0, deadlines >= 1, slots >= 1.")
    out = {}
    for mode in CSP_MODES:
        csp = StudyCSP(counts, weeks, spw)
        r = csp.solve(csp_config(mode, max_steps))
        out[mode] = {"result": r, "csp": csp,
                     "timetable": csp.timetable(r.solution) if r.solution else [],
                     "valid": bool(r.solution is not None and check_solution(counts, weeks, spw, r.solution))}
    return out


def csp_summary(runs: Mapping[str, dict], counts, weeks, spw) -> pd.DataFrame:
    rows = []
    for mode, d in runs.items():
        r, st = d["result"], d["result"].stats
        sizes0 = list(r.domain_sizes_initial.values())
        sizes1 = list(r.domain_sizes_after_ac3.values()) or sizes0
        rows.append({"Mode": mode, "Status": r.status, "Valid schedule": d["valid"], "Variables (sessions)": len(sizes0),
                     "Assignments": st.assignments, "Backtracks": st.backtracks, "Consistency checks": st.checks,
                     "FC prunings": st.fc_prunings, "AC-3 prunings": st.ac3_prunings, "AC-3 revisions": st.ac3_revisions,
                     "log10 raw space (initial)": _log10_space(sizes0), "log10 raw space (after AC-3)": _log10_space(sizes1),
                     "Message": r.message})
    return pd.DataFrame(rows)


def _log10_space(sizes) -> float:
    return float(sum(math.log10(s) for s in sizes)) if sizes and all(s > 0 for s in sizes) else float("-inf")


def domain_frame(result) -> pd.DataFrame:
    init = result.domain_sizes_initial
    after = result.domain_sizes_after_ac3 or init
    return pd.DataFrame([{"Session": f"A{v[0] + 1}·S{v[1] + 1}", "Initial domain size": init[v],
                          "After AC-3": after.get(v, init[v]), "Removed by AC-3": init[v] - after.get(v, init[v])}
                         for v in init])


def find_csp_rejection_example(ctx: Context, weeks=(1, 6, 10), spw: int = 2, max_profiles: int = 300):
    """Find a real case where the NN says 'target reached' but the CSP rejects the schedule.
    Same recipe as run_demo.py DEMO 3: tight deadlines + 2 slots/week, UCS, goal mean then each."""
    weeks = list(weeks)
    for p in make_profiles(ctx.test, max_profiles, ctx.seed):
        q = dataclasses.replace(p, weeks_avail=weeks)
        tgt = target_rule(ctx.nn.marks_table(p))
        for goal in ("mean", "each"):
            r = plan(q, ctx.nn, tgt, spw, "ucs", goal)
            rejected = [t for t in r.trace if t["rejected"]]
            if rejected:
                s = rejected[0]["state"]
                marks = [r.table[i][s[i]] for i in range(len(s))]
                sc = Scenario(p.student_id, tgt, spw, goal, tuple(weeks), True)
                return {"scenario": sc, "profile": q, "result": r, "state": s, "marks": marks, "n_rejected": len(rejected),
                        "capacity": capacity_report(s, weeks, spw), "csp_ok": capacity_ok(s, weeks, spw),
                        "runs": run_csp_modes(s, weeks, spw)}
    return None


# --------------------------------------------------------------------------------------- NN
def nn_live_metrics(ctx: Context) -> pd.DataFrame:
    """Held-out metrics of the LOADED model vs baselines (eval.metrics.nn_vs_baselines)."""
    d = nn_vs_baselines(ctx.model, ctx.train, ctx.test)
    return pd.DataFrame([{"Model": k, **v} for k, v in d.items()])


def nn_prediction_frame(ctx: Context) -> pd.DataFrame:
    X = ctx.test[FEATURES].values
    return pd.DataFrame({"Actual": ctx.test[TARGET].values, "NN": ctx.model.predict(to_model_input(X)),
                         "Average baseline": AvgMarkBaseline.fit(ctx.train).predict_raw(X)})


def whatif_compare(ctx: Context, profile: StudentProfile) -> pd.DataFrame:
    rows = []
    for pred in (ctx.nn, ctx.avg):
        for i, row in enumerate(pred.marks_table(profile)):
            rows += [{"Predictor": pred.name, "Assignment": f"A{i + 1}", "Sessions": k, "Predicted mark": v}
                     for k, v in enumerate(row)]
    return pd.DataFrame(rows)


@functools.lru_cache(maxsize=4)
def train_fresh_nn(ctx_seed: int, seed: int):
    """OPTIONAL, user-triggered: fit a new MLP in memory (not saved) to expose training curves."""
    ctx = load_context(ctx_seed)
    m = fit_model(ctx.train, seed=seed)
    hist = pd.DataFrame(m.history, columns=["train_mse", "val_mse"])
    hist.insert(0, "Epoch", np.arange(1, len(hist) + 1))
    hist["Training RMSE (marks)"] = np.sqrt(hist.train_mse) * 100      # history is MSE in (mark/100)^2 units
    hist["Validation RMSE (marks)"] = np.sqrt(hist.val_mse) * 100
    mt = pd.DataFrame([{"Model": k, **v} for k, v in nn_vs_baselines(m, ctx.train, ctx.test).items()])
    return hist, mt


def monotonicity_live(ctx: Context, n: int = 30) -> dict:
    return monotonicity_report(ctx.model, ctx.profiles[:n])


# --------------------------------------------------------------------------------------- experiments
RESULT_FILES = {"nn": "nn_results.csv", "exp1": "exp1_search.csv", "exp2": "exp2_csp.csv",
                "exp3": "exp3_nn_vs_avg.csv", "mono": "monotonicity.csv"}


def load_results(results_dir: Path | str = RESULTS_DIR) -> dict:
    """Read the stored experiment tables. Missing files -> None (the page shows how to generate them)."""
    out = {}
    for k, fn in RESULT_FILES.items():
        p = Path(results_dir) / fn
        try:
            out[k] = pd.read_csv(p) if p.exists() else None
        except Exception:      # corrupted file: treat as missing, never crash the GUI
            out[k] = None
    return out


def stored_seed_count(results: Mapping) -> int | None:
    """The CSVs hold mean/std only; the number of seeds is read from monotonicity.csv (one row per seed)."""
    m = results.get("mono")
    return None if m is None else int(len(m))


@functools.lru_cache(maxsize=2)
def exp1_with_generated(ctx_seed: int, n_seeds: int = 5, n_profiles: int = 30, goal: str = "each") -> tuple:
    """Experiment 1 re-run live with the GENERATED-node counter, which results/exp1_search.csv does not contain.
    Mirrors eval.experiments.exp1_search (same seeds, profiles, target rule); only the extra column is new."""
    ctx = load_context(ctx_seed)
    rows = []
    for s in range(n_seeds):
        nn = NNPredictor(fit_model(ctx.train, seed=s))
        profs = make_profiles(ctx.test, n_profiles, s)
        for algo in ALGOS:
            exp = gen = 0.0; costs = []; solved = 0
            for p in profs:
                r = plan(p, nn, target_rule(nn.marks_table(p)), SLOTS_PER_WEEK, algo, goal)
                exp += r.search.expanded; gen += r.search.generated
                if r.status == "OK":
                    solved += 1; costs.append(r.cost)
            n = len(profs)
            rows.append({"seed": s, "algo": algo, "expanded": exp / n, "generated": gen / n,
                         "cost": float(np.mean(costs)) if costs else np.nan, "solved_rate": solved / n})
    per_seed = pd.DataFrame(rows)
    g = per_seed.groupby("algo")[["expanded", "generated", "cost", "solved_rate"]].agg(["mean", "std"])
    g.columns = [f"{c}_{s}" for c, s in g.columns]
    return per_seed, g.reset_index().round(3)


@functools.lru_cache(maxsize=8)
def cached_rejection_example(ctx_seed: int, weeks: tuple = (1, 6, 10), spw: int = 2):
    """lru_cache wrapper (the search over profiles takes ~0.3 s)."""
    return find_csp_rejection_example(load_context(ctx_seed), weeks, spw)


STRESS_EXAMPLE = {"counts": (4, 5, 8), "weeks": (3, 5, 8), "spw": 2}   # illustrative infeasible instance (BT blows up)


def mean_std_table(df: pd.DataFrame, keys: list, metrics: list, nd: int = 2) -> pd.DataFrame:
    """Collapse '<m>_mean' and '<m>_std' columns of a stored summary into display strings 'mean ± std'."""
    out = df[keys].copy()
    for m in metrics:
        out[m] = [f"{a:.{nd}f} ± {b:.{nd}f}" for a, b in zip(df[f"{m}_mean"], df[f"{m}_std"])]
    return out
