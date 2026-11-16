"""GUI smoke tests.
* Real Streamlit installed  -> Streamlit's AppTest runs app.py headlessly and drives every page.
* Streamlit/Plotly missing  -> a strict fake (tests/gui_stubs.py) runs every page's code with real project data, so name errors,
  bad widget ranges, duplicate keys and data-shape bugs are still caught (it cannot validate Streamlit API semantics)."""
import importlib, runpy, sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
PAGES = ["📊 Study Planner Dashboard", "🔎 Search Algorithms", "🧩 CSP Scheduling", "🧠 Neural Network", "🧪 Experiments & Results"]
try:
    import streamlit, plotly   # noqa: F401
    REAL = hasattr(streamlit, "testing")
except Exception:
    REAL = False


@pytest.mark.skipif(not REAL, reason="real streamlit/plotly not installed")
@pytest.mark.parametrize("page", PAGES)
def test_app_pages_with_apptest(page):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180).run()
    assert not at.exception
    at.sidebar.radio(key="nav").set_value(page).run()
    assert not at.exception
    if page == PAGES[0]:
        [b for b in at.button if b.label == "Generate Study Plan"][0].click().run()
        assert not at.exception and not at.error


@pytest.mark.skipif(REAL, reason="real streamlit available: AppTest is used instead")
@pytest.mark.parametrize("nn_on", [True, False])
@pytest.mark.parametrize("page_index", range(5))
def test_pages_with_fake_streamlit(page_index, nn_on):
    from tests import gui_stubs
    saved = {k: sys.modules.get(k) for k in ["streamlit", "plotly", "plotly.graph_objects", "plotly.subplots"]}
    try:
        st = gui_stubs.install()
        from gui import helpers as h
        from gui.views import csp_page, dashboard, experiments_page, nn_page, search_page
        ctx = h.load_context()
        views = [dashboard, search_page, csp_page, nn_page, experiments_page]
        st.press = True                                  # every button "clicked"
        st.reset(); views[page_index].render(ctx, nn_on)
        if page_index == 2:                              # also the other CSP sources
            for src in ["Custom session counts", "Stress example (infeasible, shows pruning)"]:
                st.session_state["csp_source"] = src; st.reset(); views[2].render(ctx, nn_on)
            for mode in list(h.CSP_MODES):
                st.session_state["csp_mode"] = mode; st.reset(); views[2].render(ctx, nn_on)
        if page_index == 1:
            for algo in h.ALGO_LABELS:
                st.session_state["search_algo"] = algo; st.reset(); views[1].render(ctx, nn_on)
        assert st.chart_keys                      # every page drew at least one chart
    finally:
        for k, v in saved.items():
            if v is None: sys.modules.pop(k, None)
            else: sys.modules[k] = v
        for m in [m for m in sys.modules if m == "gui" or (m.startswith("gui.") and m != "gui.helpers")]:
            sys.modules.pop(m, None)


@pytest.mark.skipif(REAL, reason="real streamlit available: AppTest is used instead")
def test_app_script_with_fake_streamlit():
    from tests import gui_stubs
    saved = {k: sys.modules.get(k) for k in ["streamlit", "plotly", "plotly.graph_objects", "plotly.subplots"]}
    try:
        st = gui_stubs.install(); st.press = True
        for page in PAGES:
            st.session_state["nav"] = page; st.reset()
            runpy.run_path(str(ROOT / "app.py"), run_name="__main__")
    finally:
        for k, v in saved.items():
            if v is None: sys.modules.pop(k, None)
            else: sys.modules[k] = v
        for m in [m for m in sys.modules if m == "gui" or (m.startswith("gui.") and m != "gui.helpers")]:
            sys.modules.pop(m, None)
