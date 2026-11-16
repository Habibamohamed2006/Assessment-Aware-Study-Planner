"""Shared read-only scenario strip for pages 2-3 (the scenario itself is edited on the Dashboard)."""
from __future__ import annotations

import streamlit as st

from gui import helpers as h, ui


def current(ctx, nn_on: bool) -> h.Scenario:
    sc = h.resolve_scenario(ctx, st.session_state, nn_on)
    ui.chips([f"Student {sc.student_id}", f"Target {sc.target:g}", f"{sc.spw} slots/week", f"Deadlines (weeks) {list(sc.weeks)}",
              f"Goal: {sc.goal}", f"Predictor: {'NN' if nn_on else 'Average baseline'}"])
    st.caption("Scenario comes from the Dashboard page: change student, target, slots, deadlines or the NN switch there / in the sidebar.")
    return sc
