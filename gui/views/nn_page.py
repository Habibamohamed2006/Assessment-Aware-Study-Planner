"""Page 4 - Neural network analysis."""
from __future__ import annotations

import streamlit as st

from gui import charts, helpers as h, ui
from gui.views import scenario


def render(ctx, nn_on: bool) -> None:
    ui.page_header("Neural network analysis", "The NN predicts an assignment mark from 12 behavioural/assessment features; "
                   "extra study sessions enter as extra clicks (project rule: 1 session = +100 clicks).")
    sc = scenario.current(ctx, nn_on)
    profile = ctx.profile(sc.student_id)
    n_train, n_test = len(ctx.train), len(ctx.test)
    c = st.columns(4)
    ui.metric_card(c[0], "Predictor in use", "NN ON" if nn_on else "NN OFF", "trained MLP" if nn_on else "average-mark baseline", "accent" if nn_on else "")
    ui.metric_card(c[1], "Architecture", "12-16-8-1", "tanh MLP, Adam, early stopping")
    ui.metric_card(c[2], "Training rows", f"{n_train:,}", f"presentations {', '.join(ctx.order[:-1])}")
    ui.metric_card(c[3], "Held-out rows", f"{n_test:,}", f"presentation {ctx.order[-1]}")
    st.write("")
    with st.expander("NN ON vs NN OFF: what is the difference?", expanded=False):
        st.markdown(
            "- **NN ON** uses the saved trained network (`models/…npz`, loaded, never retrained). Its prediction depends on the "
            "student's clicks, so adding sessions (extra clicks) can raise the predicted mark and the planner can allocate effort.\n"
            "- **NN OFF** uses the project's *average-mark baseline*: the mean of the student's previous marks (or the training "
            "mean). It ignores effort completely, so every extra session predicts the same mark.\n"
            "- Switch with the sidebar toggle; it applies to the Dashboard, Search and CSP pages. This page always shows both.")
    t1, t2, t3, t4 = st.tabs(["Predicted marks", "Model performance", "Actual vs predicted", "Training & reliability"])

    with t1:
        st.markdown(f"##### Student {sc.student_id}: predicted marks after k extra sessions")
        both = h.whatif_compare(ctx, profile)
        zero = both[both.Sessions == 0].pivot(index="Assignment", columns="Predictor", values="Predicted mark").round(1)
        eight = both[both.Sessions == 8].pivot(index="Assignment", columns="Predictor", values="Predicted mark").round(1)
        summary = zero.join(eight, lsuffix=" (0 sessions)", rsuffix=" (8 sessions)")
        summary["Actual (hidden)"] = [round(x, 1) for x in profile.true_scores]
        ui.show_table(summary.reset_index())
        a, b = st.columns(2)
        for col, pname, key in [(a, "NN", "nn_whatif"), (b, "AvgMark", "avg_whatif")]:
            with col:
                ui.show_chart(charts.whatif_curves(both[both.Predictor == pname].drop(columns="Predictor"), sc.target, None,
                                                   f"{'NN' if pname == 'NN' else 'Average baseline (NN OFF)'}"), key)
        st.caption("The NN curve slopes upward (effort matters, enforced to be non-decreasing by a running maximum). "
                   "The baseline is flat: it cannot tell whether study helps.")

    with t2:
        live = h.nn_live_metrics(ctx)
        st.markdown("##### Loaded model on the held-out presentation (computed now)")
        ui.show_table(live.round(3))
        cols = st.columns(3)
        for col, m in zip(cols, ["MAE", "RMSE", "R2"]):
            with col:
                ui.show_chart(charts.metric_compare(live, m), f"nnm_{m}")
        st.caption("Lower MAE/RMSE and higher R² are better.")
        res = h.load_results()
        if res["nn"] is not None:
            k = h.stored_seed_count(res)
            st.markdown(f"##### Stored evaluation across seeds{f' (n = {k})' if k else ''}: mean ± std")
            ui.show_table(h.mean_std_table(res["nn"], ["model"], ["MAE", "RMSE", "R2"], 3))
        else:
            st.info("results/nn_results.csv not found. Create it with `python -m eval.experiments --seeds 5 --goal each`.")

    with t3:
        df = h.nn_prediction_frame(ctx)
        a, b = st.columns(2)
        with a:
            ui.show_chart(charts.actual_vs_pred(df, "NN", "NN", charts.PURPLE), "avp_nn")
        with b:
            ui.show_chart(charts.actual_vs_pred(df, "Average baseline", "Average baseline", charts.AMBER), "avp_avg")
        ui.show_chart(charts.residual_hist(df), "resid")
        st.caption("Scatter is a fixed-seed random sample for speed; all metrics use every held-out row.")

    with t4:
        st.markdown("##### Training curves")
        st.write("The saved model file stores weights only, not its training history, so its curve cannot be shown. "
                 "You can fit a fresh model in memory (about 3 s, nothing is saved or replaced) to see a real training curve.")
        seed = st.number_input("Seed for the fresh fit", 0, 10_000, 0, key="fresh_seed")
        if st.button("Train a fresh model for the curve"):
            with st.spinner("Training…"):
                st.session_state["fresh"] = (int(seed), *h.train_fresh_nn(ctx.seed, int(seed)))
        fr = st.session_state.get("fresh")
        if fr:
            ui.show_chart(charts.training_curve(fr[1]), "train_curve")
            st.caption(f"Seed {fr[0]}: stopped after {len(fr[1])} epochs (early stopping on a validation split). Held-out metrics of this fresh model:")
            ui.show_table(fr[2].round(3))
        st.markdown("##### Monotonicity (do more sessions ever lower the raw NN prediction?)")
        live_m = h.monotonicity_live(ctx)
        st.json(live_m)
        res = h.load_results()
        if res["mono"] is not None:
            with st.expander("Stored per-seed monotonicity results"):
                ui.show_table(res["mono"])
        st.caption("The planner applies a running maximum, so its what-if tables never decrease. Raw violations are reported honestly.")
