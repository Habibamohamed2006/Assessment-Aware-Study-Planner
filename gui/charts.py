"""Plotly figure builders. Pure functions: DataFrames in, figures out (no Streamlit import)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

NAVY, PURPLE, SOFT, TEAL, AMBER, RED, GREY, INK = "#14213D", "#7C5CFC", "#C9BEFF", "#1F9D8B", "#E8A33D", "#D64550", "#8C96A8", "#24304A"
ALGO_COLORS = {"ucs": PURPLE, "greedy": AMBER, "astar": TEAL}
ALGO_NAMES = {"ucs": "UCS", "greedy": "Greedy", "astar": "A*"}
CSP_COLORS = {"Backtracking only": GREY, "Backtracking + Forward Checking": PURPLE,
              "Backtracking + Forward Checking + AC-3": NAVY}
SHORT_CSP = {"Backtracking only": "BT", "Backtracking + Forward Checking": "BT+FC",
             "Backtracking + Forward Checking + AC-3": "BT+FC+AC-3", "BT": "BT", "BT+FC": "BT+FC", "BT+FC+AC3": "BT+FC+AC-3"}
ASSIGN_COLORS = [PURPLE, TEAL, AMBER, "#4C8DFF", "#E26BA6", "#7BC043"]


def _base(fig: go.Figure, title: str | None = None, height: int = 360) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, x=0, font=dict(size=15, color=NAVY)) if title else None,
        height=height, margin=dict(l=10, r=10, t=48 if title else 16, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, Segoe UI, system-ui, sans-serif", size=12, color=INK),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hoverlabel=dict(font_size=12))
    fig.update_xaxes(gridcolor="#E4E7F0", zeroline=False, linecolor="#CBD0DE")
    fig.update_yaxes(gridcolor="#E4E7F0", zeroline=False, linecolor="#CBD0DE")
    return fig


# ------------------------------------------------------------------ planner / dashboard
def marks_vs_target(table: pd.DataFrame) -> go.Figure:
    """table = helpers.per_assignment_table(): baseline vs planned predicted marks with the target line."""
    fig = go.Figure()
    fig.add_bar(x=table["Assignment"], y=table["Predicted (no extra effort)"], name="No extra effort", marker_color=SOFT)
    fig.add_bar(x=table["Assignment"], y=table["Predicted (with plan)"], name="With plan", marker_color=PURPLE,
                text=table["Predicted (with plan)"].round(1), textposition="outside")
    fig.add_hline(y=float(table["Target"].iloc[0]), line=dict(color=RED, dash="dash"),
                  annotation_text="Target", annotation_position="top left")
    fig.update_layout(barmode="group")
    lo = max(0, min(table[["Predicted (no extra effort)", "Predicted (with plan)"]].min().min(), table["Target"].iloc[0]) - 15)
    fig.update_yaxes(range=[lo, 100], title="Predicted mark")
    return _base(fig, "Predicted marks vs target")


def sessions_bar(table: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Bar(x=table["Assignment"], y=table["Sessions"], marker_color=[ASSIGN_COLORS[i % 6] for i in range(len(table))],
                           text=table["Sessions"], textposition="outside"))
    fig.update_yaxes(title="Study sessions", rangemode="tozero", dtick=1)
    return _base(fig, "Study sessions per assignment")


def whatif_curves(long: pd.DataFrame, target: float, allocation=None, title="What-if curves: predicted mark vs sessions") -> go.Figure:
    fig = go.Figure()
    for i, (name, g) in enumerate(long.groupby("Assignment")):
        c = ASSIGN_COLORS[i % 6]
        fig.add_scatter(x=g["Sessions"], y=g["Predicted mark"], mode="lines+markers", name=name, line=dict(color=c, width=2.5),
                        marker=dict(size=6))
        if allocation is not None:
            k = int(allocation[i])
            fig.add_scatter(x=[k], y=[g["Predicted mark"].iloc[k]], mode="markers", showlegend=False,
                            marker=dict(size=15, color=c, symbol="diamond", line=dict(width=2, color=NAVY)),
                            hovertemplate=f"{name}: planned {k} sessions<extra></extra>")
    fig.add_hline(y=target, line=dict(color=RED, dash="dash"), annotation_text="Target", annotation_position="bottom right")
    fig.update_xaxes(title="Sessions on the assignment", dtick=1)
    fig.update_yaxes(title="Predicted mark")
    return _base(fig, title)


def timetable_heatmap(grid) -> go.Figure:
    """grid = helpers.TimetableGrid. Free slots are light, each assignment has its own colour."""
    n = int(grid.ids.max()) if grid.ids.size else 0
    cols = ["#EEF0F7"] + [ASSIGN_COLORS[i % 6] for i in range(max(n, 1))]
    m = max(len(cols) - 1, 1)
    scale = []
    for i, c in enumerate(cols):                    # discrete colour scale
        scale += [[i / (m + 1), c], [(i + 1) / (m + 1), c]]
    fig = go.Figure(go.Heatmap(z=grid.ids, x=grid.slot_labels, y=grid.week_labels, text=grid.labels, texttemplate="%{text}",
                               colorscale=scale, zmin=-0.5, zmax=m + 0.5, showscale=False, xgap=3, ygap=3,
                               hovertemplate="%{y}, %{x}: %{text}<extra></extra>"))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(showgrid=False, side="top")
    return _base(fig, None, height=max(220, 34 * len(grid.week_labels) + 70))


# ------------------------------------------------------------------ search
def search_metric_bars(summary: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=1, cols=3, subplot_titles=("Nodes expanded", "Nodes generated", "Solution cost (sessions)"))
    for j, col in enumerate(["Expanded", "Generated", "Solution cost (sessions)"], start=1):
        fig.add_bar(x=[ALGO_NAMES[k] for k in summary["key"]], y=summary[col].fillna(0), showlegend=False, row=1, col=j,
                    marker_color=[ALGO_COLORS[k] for k in summary["key"]], text=summary[col].fillna(0), textposition="outside")
    return _base(fig, None, 320)


def search_space_3d(trace: pd.DataFrame, k_max: int, title="Expanded states in the allocation space") -> go.Figure:
    fig = go.Figure()
    exp = trace[trace["Outcome"] == "Expanded"]
    fig.add_scatter3d(x=exp["A1"], y=exp["A2"], z=exp["A3"], mode="markers", name="Expanded",
                      marker=dict(size=4, color=exp["Step"], colorscale=[[0, SOFT], [1, NAVY]], showscale=True,
                                  colorbar=dict(title="Step", thickness=10)),
                      text=exp["State (sessions per assignment)"], hovertemplate="%{text}<br>step %{marker.color}<extra></extra>")
    for label, color, sym in [("NN says enough → CSP rejected", RED, "x"), ("Goal (NN ok + CSP ok)", TEAL, "diamond")]:
        d = trace[trace["Outcome"] == label]
        if len(d):
            fig.add_scatter3d(x=d["A1"], y=d["A2"], z=d["A3"], mode="markers", name=label.split(" (")[0],
                              marker=dict(size=7, color=color, symbol=sym), text=d["State (sessions per assignment)"])
    ax = dict(range=[-0.3, k_max + 0.3], dtick=1, backgroundcolor="rgba(0,0,0,0)")
    fig.update_layout(scene=dict(xaxis=dict(title="A1 sessions", **ax), yaxis=dict(title="A2 sessions", **ax),
                                 zaxis=dict(title="A3 sessions", **ax)))
    return _base(fig, title, 460)


def search_slice_heatmap(trace: pd.DataFrame, k3: int, k_max: int) -> go.Figure:
    """Expansion order on the (A1, A2) grid for a fixed A3 = k3 (blank = never expanded)."""
    z = np.full((k_max + 1, k_max + 1), np.nan)
    txt = [[""] * (k_max + 1) for _ in range(k_max + 1)]
    for _, r in trace[trace["A3"] == k3].iterrows():
        z[r["A2"], r["A1"]] = r["Step"]
        txt[r["A2"]][r["A1"]] = str(int(r["Step"])) + ("✗" if r["Outcome"].startswith("NN says") else "★" if r["Outcome"].startswith("Goal") else "")
    fig = go.Figure(go.Heatmap(z=z, x=list(range(k_max + 1)), y=list(range(k_max + 1)), text=txt, texttemplate="%{text}",
                               colorscale=[[0, SOFT], [1, NAVY]], xgap=2, ygap=2, colorbar=dict(title="Step", thickness=10),
                               hovertemplate="A1=%{x}, A2=%{y}<br>expanded at step %{z}<extra></extra>"))
    fig.update_xaxes(title="A1 sessions", dtick=1, showgrid=False)
    fig.update_yaxes(title="A2 sessions", dtick=1, showgrid=False)
    return _base(fig, f"Order of expansion, slice A3 = {k3}  (★ goal, ✗ CSP-rejected)", 420)


def search_progress(trace: pd.DataFrame, upto: int) -> go.Figure:
    t = trace[trace["Step"] <= upto]
    fig = go.Figure()
    fig.add_scatter(x=t["Step"], y=t["g (sessions so far)"], mode="lines+markers", name="g (sessions so far)", line=dict(color=PURPLE))
    fig.add_scatter(x=t["Step"], y=t["Min mark"], mode="lines", name="Min predicted mark", yaxis="y2", line=dict(color=TEAL))
    fig.update_layout(yaxis2=dict(overlaying="y", side="right", title="Min predicted mark", showgrid=False))
    fig.update_xaxes(title="Expansion step")
    fig.update_yaxes(title="g")
    return _base(fig, "Progress of the search up to the selected step", 300)


# ------------------------------------------------------------------ CSP
def csp_domain_bars(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_bar(x=df["Session"], y=df["After AC-3"], name="Left after AC-3", marker_color=NAVY)
    fig.add_bar(x=df["Session"], y=df["Removed by AC-3"], name="Pruned by AC-3", marker_color=RED)
    fig.update_layout(barmode="stack")
    fig.update_yaxes(title="Domain size (slots)")
    return _base(fig, "Domains per session variable: kept vs pruned by AC-3", 320)


def csp_mode_comparison(summary: pd.DataFrame) -> go.Figure:
    metrics = [("Assignments", "Assignments"), ("Backtracks", "Backtracks"), ("Consistency checks", "Consistency checks")]
    fig = make_subplots(rows=1, cols=3, subplot_titles=[m[0] for m in metrics])
    modes = list(summary["Mode"])
    for j, (col, _) in enumerate(metrics, start=1):
        fig.add_bar(x=[SHORT_CSP[m] for m in modes], y=summary[col], showlegend=False, row=1, col=j,
                    marker_color=[CSP_COLORS[m] for m in modes], text=summary[col], textposition="outside")
    fig.update_yaxes(type="log", row=1, col=3)
    return _base(fig, None, 320)


def csp_pruning_bars(summary: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_bar(x=[SHORT_CSP[m] for m in summary["Mode"]], y=summary["FC prunings"], name="Forward-checking prunings", marker_color=PURPLE)
    fig.add_bar(x=[SHORT_CSP[m] for m in summary["Mode"]], y=summary["AC-3 prunings"], name="AC-3 prunings", marker_color=TEAL)
    fig.update_layout(barmode="group")
    fig.update_yaxes(title="Values removed from domains")
    return _base(fig, "Domain values removed", 320)


def capacity_chart(cap: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    labels = [f"{a} (due wk {w})" for a, w in zip(cap["Assignment"], cap["Deadline (weeks)"])]
    fig.add_bar(x=labels, y=cap["Slots available by then"], name="Slots available", marker_color=SOFT)
    fig.add_bar(x=labels, y=cap["Sessions due by then (cumulative)"], name="Sessions due (cumulative)",
                marker_color=[TEAL if ok else RED for ok in cap["Enough slots"]])
    fig.update_layout(barmode="group")
    fig.update_yaxes(title="Slots / sessions")
    return _base(fig, "Capacity check at each deadline", 320)


# ------------------------------------------------------------------ NN
def actual_vs_pred(df: pd.DataFrame, col: str, name: str, color: str, n: int = 3000, seed: int = 0) -> go.Figure:
    d = df.sample(min(n, len(df)), random_state=seed)
    fig = go.Figure()
    fig.add_scatter(x=d["Actual"], y=d[col], mode="markers", name=name, marker=dict(size=4, color=color, opacity=0.35))
    fig.add_scatter(x=[0, 100], y=[0, 100], mode="lines", name="Perfect", line=dict(color=RED, dash="dash"))
    fig.update_xaxes(title="Actual mark", range=[0, 100])
    fig.update_yaxes(title="Predicted mark", range=[0, 100])
    return _base(fig, f"Actual vs predicted: {name} ({len(d):,} of {len(df):,} held-out rows)", 380)


def residual_hist(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for col, c in [("NN", PURPLE), ("Average baseline", AMBER)]:
        fig.add_histogram(x=df[col] - df["Actual"], name=col, marker_color=c, opacity=0.6, nbinsx=50)
    fig.update_layout(barmode="overlay")
    fig.update_xaxes(title="Prediction error (predicted − actual, marks)")
    fig.update_yaxes(title="Rows")
    return _base(fig, "Error distribution on the held-out presentation", 340)


def metric_compare(df: pd.DataFrame, metric: str) -> go.Figure:
    fig = go.Figure(go.Bar(x=df["Model"], y=df[metric], text=df[metric].round(3), textposition="outside",
                           marker_color=[PURPLE if "NN" in m else GREY for m in df["Model"]]))
    fig.update_yaxes(title=metric, rangemode="tozero")
    return _base(fig, metric, 300)


def training_curve(hist: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_scatter(x=hist["Epoch"], y=hist["Training RMSE (marks)"], name="Training", line=dict(color=PURPLE))
    fig.add_scatter(x=hist["Epoch"], y=hist["Validation RMSE (marks)"], name="Validation", line=dict(color=TEAL))
    fig.update_xaxes(title="Epoch")
    fig.update_yaxes(title="RMSE (marks)")
    return _base(fig, "Training curve (fresh in-session fit; early stopping applies)", 340)


# ------------------------------------------------------------------ experiments
def mean_std_bars(df: pd.DataFrame, x: str, group: str | None, mean_col: str, std_col: str, title: str, ytitle: str,
                  colors: dict | None = None, names: dict | None = None, log: bool = False) -> go.Figure:
    fig = go.Figure()
    groups = [None] if group is None else list(dict.fromkeys(df[group]))
    for g in groups:
        d = df if g is None else df[df[group] == g]
        xs = [str(v) for v in d[x]]
        color = (colors or {}).get(g, PURPLE) if g is not None else [(colors or {}).get(v, PURPLE) for v in d[x]]
        fig.add_bar(x=[(names or {}).get(v, v) for v in xs] if g is None else xs, y=d[mean_col], name=str((names or {}).get(g, g)),
                    marker_color=color, error_y=dict(type="data", array=d[std_col], visible=True, color=NAVY),
                    showlegend=g is not None)
    fig.update_layout(barmode="group")
    fig.update_yaxes(title=ytitle, type="log" if log else "linear")
    return _base(fig, title, 340)
