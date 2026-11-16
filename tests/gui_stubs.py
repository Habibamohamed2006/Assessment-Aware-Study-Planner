"""Minimal fake `streamlit` and `plotly` used to smoke-run the page code where the real packages are not installed.
The fake enforces the Streamlit rules that most often break apps: unique widget keys, unique chart keys, valid
number_input / slider / select ranges, and values taken from options."""
import sys, types
from contextlib import contextmanager


class SS(dict):
    __getattr__ = dict.get
    def __setattr__(self, k, v): self[k] = v


class Rec:
    """Plotly stand-in: accepts any attribute/call/subscript and stays a Rec."""
    def __init__(self, *a, **k): pass
    def __getattr__(self, name): return Rec()
    def __call__(self, *a, **k): return Rec()
    def __iter__(self): return iter(())


class Node:
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def __getattr__(self, name): return getattr(FAKE, name)


class StopApp(Exception):
    pass


def _unique(key):
    if key is not None:
        assert key not in FAKE.keys_run, f"DuplicateWidgetID {key}"
        FAKE.keys_run.add(key)


def build():
    f = types.ModuleType("streamlit"); f.session_state = SS(); f.press = False
    f.keys_run, f.chart_keys, f.log = set(), set(), []
    f.sidebar = Node()
    def reset(): f.keys_run.clear(); f.chart_keys.clear(); f.log.clear()
    f.reset = reset
    def out(*a, **k): f.log.append((a[:1], k.get("key"))); return None
    for n in ["markdown", "write", "caption", "error", "success", "info", "warning", "json", "dataframe", "image", "subheader",
              "title", "header", "exception", "set_page_config", "rerun"]:
        setattr(f, n, out)
    def stop(): raise StopApp()
    f.stop = stop
    def _val(key, default):
        if key is not None and key in f.session_state: return f.session_state[key]
        if key is not None: f.session_state[key] = default
        return default
    def selectbox(label, options, index=0, key=None, format_func=str, **k):
        _unique(key); options = list(options); v = _val(key, options[index] if options else None)
        assert v in options, f"{label}: value {v!r} not in options"; [format_func(o) for o in options[:3]]; return v
    f.selectbox = f.radio = selectbox
    def select_slider(label, options=(), value=None, key=None, **k):
        _unique(key); options = list(options); v = _val(key, value if value is not None else options[0]); assert v in options; return v
    f.select_slider = select_slider
    def slider(label, min_value, max_value, value=None, key=None, **k):
        _unique(key); assert min_value < max_value, f"slider {label}: min must be < max"
        v = _val(key, value if value is not None else min_value); assert min_value <= v <= max_value; return v
    f.slider = slider
    def number_input(label, min_value=None, max_value=None, value=None, step=None, key=None, **k):
        _unique(key); v = _val(key, value if value is not None else min_value)
        if min_value is not None: assert v >= min_value, f"{label}: {v} < min {min_value}"
        if max_value is not None: assert v <= max_value, f"{label}: {v} > max {max_value}"
        if isinstance(min_value, float) or isinstance(max_value, float): assert isinstance(v, float), f"{label}: mixed int/float"
        return v
    f.number_input = number_input
    def toggle(label, value=False, key=None, **k): _unique(key); return _val(key, value)
    f.toggle = f.checkbox = toggle
    def multiselect(label, options, default=None, key=None, **k): _unique(key); return _val(key, list(default or []))
    f.multiselect = multiselect
    def button(label, key=None, on_click=None, args=(), **k):
        _unique(key if key else "btn:" + label)
        if f.press and on_click: on_click(*args)
        return bool(f.press)
    f.button = button
    def columns(spec, **k): return [Node() for _ in range(spec if isinstance(spec, int) else len(spec))]
    f.columns = columns
    f.tabs = lambda labels, **k: [Node() for _ in labels]
    f.expander = lambda *a, **k: Node()
    f.spinner = lambda *a, **k: Node()
    f.container = lambda *a, **k: Node()
    def plotly_chart(fig, key=None, **k):
        assert key is not None, "plotly_chart without key"; assert key not in f.chart_keys, f"duplicate chart key {key}"
        f.chart_keys.add(key)
    f.plotly_chart = plotly_chart
    return f


FAKE = None


def install(real_ok: bool = False):
    """Install the fakes into sys.modules (idempotent). Returns the fake streamlit module."""
    global FAKE
    FAKE = build(); sys.modules["streamlit"] = FAKE
    go = types.ModuleType("plotly.graph_objects"); go.__getattr__ = lambda name: Rec
    sp = types.ModuleType("plotly.subplots"); sp.make_subplots = lambda *a, **k: Rec()
    pl = types.ModuleType("plotly"); pl.graph_objects, pl.subplots = go, sp
    sys.modules.update({"plotly": pl, "plotly.graph_objects": go, "plotly.subplots": sp})
    for m in [m for m in sys.modules if m.startswith("gui.") and m != "gui.helpers"] + ["gui"]:
        sys.modules.pop(m, None)
    return FAKE
