"""Streamlit GUI for the Assessment-Aware Study Planner.   Launch:  streamlit run app.py
All computation is delegated to the project's modules through gui/helpers.py; this file only wires up navigation."""
import streamlit as st

from gui import helpers as h, ui

PERSIST = ("sel_", "target_", "weeks_", "plan_", "search_algo", "csp_", "rej_", "e2_", "nn_on", "nav")


def keep_widget_state() -> None:
    """Streamlit drops the state of widgets that are not drawn on the current page; re-assign to keep it across pages."""
    for k in list(st.session_state.keys()):
        if k.startswith(PERSIST):
            st.session_state[k] = st.session_state[k]


def main() -> None:
    st.set_page_config(page_title="Assessment-Aware Study Planner", page_icon="🎓", layout="wide")
    ui.inject_css()
    keep_widget_state()
    try:
        ctx = h.load_context()
    except h.GuiError as e:
        st.error(str(e))
        st.stop()
        return
    except Exception as e:
        st.error(f"Could not load the project data or model: {e}")
        st.stop()
        return

    with st.sidebar:
        st.markdown("## 🎓 Study Planner")
        st.caption("Assessment-aware planning with NN + Search + CSP")
        page = st.radio("Navigate", ui.PAGES, key="nav", label_visibility="collapsed")
        st.markdown("---")
        nn_on = st.toggle("Neural network ON", value=True, key="nn_on",
                          help="ON: trained NN predicts marks. OFF: average-of-previous-marks baseline (ignores effort).")
        st.caption("Predictor: " + ("trained neural network" if nn_on else "average-mark baseline"))
        st.markdown("---")
        st.caption(f"Data: {len(ctx.df):,} rows · test presentation {ctx.order[-1]}\n\nModel: {ctx.model_path.name}")

    from gui.views import csp_page, dashboard, experiments_page, nn_page, search_page
    views = dict(zip(ui.PAGES, [dashboard, search_page, csp_page, nn_page, experiments_page]))
    try:
        views[page].render(ctx, nn_on)
    except h.GuiError as e:
        st.error(str(e))
    except Exception as e:  # last-resort guard: show a readable message instead of a stack trace
        st.error(f"Something went wrong while rendering this page: {e}")
        st.exception(e)


main()
