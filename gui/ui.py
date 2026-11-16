"""Streamlit presentation helpers: theme (CSS), metric cards, version-tolerant wrappers."""
from __future__ import annotations

import html
import inspect

import streamlit as st

CSS = """
<style>
:root{--navy:#14213D;--navy2:#1D2D52;--purple:#7C5CFC;--soft:#EDE9FF;--bg:#F5F6FB;--card:#fff;--line:#E4E7F0;--muted:#5B6685;}
.block-container{padding-top:1.6rem;padding-bottom:3rem;max-width:1280px}
h1,h2,h3{color:var(--navy);letter-spacing:-.01em}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#14213D 0%,#1D2D52 100%)}
[data-testid="stSidebar"] *{color:#E8EBF7}
[data-testid="stSidebar"] hr{border-color:rgba(255,255,255,.15)}
[data-testid="stSidebar"] [role="radiogroup"] label{padding:.35rem .5rem;border-radius:8px}
[data-testid="stSidebar"] [role="radiogroup"] label:hover{background:rgba(255,255,255,.08)}
.hero{background:linear-gradient(120deg,#14213D 0%,#2B2F77 60%,#7C5CFC 130%);color:#fff;padding:1.5rem 1.7rem;border-radius:16px;margin-bottom:1.2rem}
.hero h1{color:#fff;margin:0 0 .3rem 0;font-size:1.9rem}
.hero p{color:#D9DDF5;margin:0;max-width:70ch;line-height:1.5}
.pagehead h2{margin-bottom:.1rem}
.pagehead p{color:var(--muted);margin-top:0}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:.95rem 1.1rem;height:100%;box-shadow:0 1px 2px rgba(20,33,61,.04)}
.card .lbl{font-size:.74rem;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);font-weight:600}
.card .val{font-size:1.65rem;font-weight:700;color:var(--navy);line-height:1.25}
.card .sub{font-size:.8rem;color:var(--muted)}
.card.good{border-left:5px solid #1F9D8B}.card.bad{border-left:5px solid #D64550}.card.accent{border-left:5px solid var(--purple)}
.chip{display:inline-block;background:var(--soft);color:#3D2FA3;border-radius:999px;padding:.15rem .7rem;font-size:.78rem;margin:0 .35rem .3rem 0;font-weight:600}
.note{background:#fff;border:1px dashed #C9BEFF;border-radius:12px;padding:.7rem 1rem;color:var(--muted);font-size:.88rem}
.banner{border-radius:12px;padding:.9rem 1.1rem;margin:.4rem 0 1rem 0;font-weight:500}
.banner.ok{background:#E4F6F2;border:1px solid #9AD8CC;color:#0E5A4D}
.banner.fail{background:#FDECEE;border:1px solid #F0AAB0;color:#8A1F2A}
.banner.warn{background:#FFF5E1;border:1px solid #F0D08A;color:#7A5200}
</style>
"""

PAGES = ["📊 Study Planner Dashboard", "🔎 Search Algorithms", "🧩 CSP Scheduling", "🧠 Neural Network", "🧪 Experiments & Results"]


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def esc(x) -> str:
    return html.escape(str(x))


def page_header(title: str, text: str) -> None:
    st.markdown(f'<div class="pagehead"><h2>{esc(title)}</h2><p>{esc(text)}</p></div>', unsafe_allow_html=True)


def hero(title: str, text: str) -> None:
    st.markdown(f'<div class="hero"><h1>{esc(title)}</h1><p>{esc(text)}</p></div>', unsafe_allow_html=True)


def metric_card(col, label, value, sub="", tone="") -> None:
    col.markdown(f'<div class="card {tone}"><div class="lbl">{esc(label)}</div><div class="val">{esc(value)}</div>'
                 f'<div class="sub">{esc(sub)}</div></div>', unsafe_allow_html=True)


def banner(kind: str, text: str) -> None:
    st.markdown(f'<div class="banner {kind}">{esc(text)}</div>', unsafe_allow_html=True)


def note(text: str) -> None:
    st.markdown(f'<div class="note">{esc(text)}</div>', unsafe_allow_html=True)


def chips(items) -> None:
    st.markdown("".join(f'<span class="chip">{esc(i)}</span>' for i in items), unsafe_allow_html=True)


def _accepts(fn, name: str) -> bool:
    try:
        return name in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


def show_chart(fig, key: str | None = None) -> None:
    """st.plotly_chart across Streamlit versions (`width='stretch'` replaced `use_container_width`)."""
    kw = {"width": "stretch"} if _accepts(st.plotly_chart, "width") else {"use_container_width": True}
    if key:
        kw["key"] = key
    st.plotly_chart(fig, **kw)


def show_table(df, **kw) -> None:
    kw.update({"width": "stretch"} if _accepts(st.dataframe, "width") else {"use_container_width": True})
    kw.setdefault("hide_index", True)
    st.dataframe(df, **kw)


def fmt_float(v, nd=1) -> str:
    return "—" if v is None else f"{v:.{nd}f}"
