"""Plots: cooperation of NN + Search + CSP, and experiment bar charts."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
from csp import StudyCSP, CSPConfig


def plot_cooperation(res, profile, path, title=""):
    fig, ax = plt.subplots(2, 2, figsize=(13, 9)); ax = ax.ravel()
    t = np.array(res.table)
    for i, row in enumerate(t):
        ax[0].plot(range(len(row)), row, marker="o", label=f"assignment {i + 1}")
    ax[0].axhline(res.target, color="k", ls="--", label=f"target {res.target:.0f}")
    if res.allocation:
        for i, k in enumerate(res.allocation):
            ax[0].scatter([k], [t[i][k]], s=140, facecolors="none", edgecolors="r")
    ax[0].set(title="NN: predicted mark vs sessions (red = chosen plan)", xlabel="sessions", ylabel="predicted mark"); ax[0].legend()
    tr = res.trace
    if tr:
        ax[1].plot([x["order"] for x in tr], [x["f"] for x in tr], ".-", label="priority of expanded node")
        rej = [x for x in tr if x["rejected"]]
        ax[1].scatter([x["order"] for x in rej], [x["f"] for x in rej], c="r", s=80, zorder=3, label="NN ok but CSP rejected")
    ax[1].set(title=f"Search trace ({res.algo.upper()}): expanded={res.search.expanded}", xlabel="expansion #", ylabel="priority"); ax[1].legend() if tr else None
    counts = res.allocation or (res.best_effort["allocation"] if res.best_effort else [])
    csp = StudyCSP(counts, res.weeks_avail, res.spw)
    if csp.vars:
        r = csp.solve(CSPConfig(False, True)); before = list(r.domain_sizes_initial.values())
        after = list(r.domain_sizes_after_ac3.values()) or before
        x = np.arange(len(before)); ax[2].bar(x - .2, before, .4, label="initial domain"); ax[2].bar(x + .2, after, .4, label="after AC-3")
        ax[2].set(title="CSP: slots left per session (AC-3 pruning)", xlabel="session variable", ylabel="#slots"); ax[2].legend()
    sched = res.schedule or (res.best_effort["schedule"] if res.best_effort else [])
    W = max(res.weeks_avail); grid = np.full((res.spw, W), np.nan)
    for a, j, w, s in sched:
        grid[s - 1, w - 1] = a + 1
    im = ax[3].imshow(grid, aspect="auto", cmap="Set2", vmin=0.5, vmax=3.5)
    for a, j, w, s in sched:
        ax[3].text(w - 1, s - 1, f"A{a + 1}.{j}", ha="center", va="center", fontsize=8)
    ax[3].set(title="Timetable (cell = assignment.session)", xlabel="week", ylabel="slot")
    ax[3].set_xticks(range(W)); ax[3].set_xticklabels(range(1, W + 1)); ax[3].set_yticks(range(res.spw)); ax[3].set_yticklabels(range(1, res.spw + 1))
    fig.suptitle(title); fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def bar_with_err(df, x, hue, y, path, title, ylabel):
    df = df.copy()
    if x == hue:
        df["_h"] = df[x]; hue = "_h"
    g = df.groupby([x, hue])[y].agg(["mean", "std"]).reset_index()
    xs = list(dict.fromkeys(g[x])); hs = list(dict.fromkeys(g[hue])); w = 0.8 / len(hs)
    fig, a = plt.subplots(figsize=(8, 4.5))
    for k, h in enumerate(hs):
        s = g[g[hue] == h].set_index(x).reindex(xs).fillna(0)
        a.bar(np.arange(len(xs)) + k * w, s["mean"], w, yerr=s["std"].fillna(0), capsize=3, label=str(h))
    a.set_xticks(np.arange(len(xs)) + w * (len(hs) - 1) / 2); a.set_xticklabels(xs)
    a.set(title=title, ylabel=ylabel, xlabel=x); a.legend(); fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)
