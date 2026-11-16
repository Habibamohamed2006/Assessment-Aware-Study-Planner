"""OULAD preprocessing: chunked aggregation of the click table -> one row per (student, assessment)."""
from pathlib import Path
import numpy as np
import pandas as pd
from config import *

CONTENT = {"resource", "oucontent", "url", "page", "subpage", "glossary", "folder", "htmlactivity", "dataplus"}
INTERACT = {"forumng", "oucollaborate", "ouwiki", "ouelluminate", "questionnaire", "dualpane"}


def activity_group(activity_type: str) -> int:
    """0 = content reading, 1 = interaction/collaboration, 2 = quizzes / homepage / other."""
    if activity_type in CONTENT:
        return 0
    if activity_type in INTERACT:
        return 1
    return 2


def build_processed(raw_dir, module: str, out_csv, chunksize: int = 500_000) -> pd.DataFrame:
    """Real pipeline (also used on the synthetic tables). Large click table is read in chunks."""
    raw = Path(raw_dir)
    required = ["assessments.csv", "studentAssessment.csv", "studentVle.csv", "vle.csv"]
    missing = [name for name in required if not (raw / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing OULAD source table(s) in {raw}: {', '.join(missing)}")
    ass = pd.read_csv(raw / "assessments.csv")
    ass = ass[(ass.code_module == module) & ass.assessment_type.isin(["TMA", "CMA"]) & ass.date.notna()].copy()
    sa = pd.read_csv(raw / "studentAssessment.csv")
    sa = sa[sa.id_assessment.isin(ass.id_assessment) & sa.score.notna()]
    vle = pd.read_csv(raw / "vle.csv", usecols=["id_site", "code_module", "activity_type"])
    vle = vle[vle.code_module == module].drop_duplicates("id_site")
    site_grp = vle.set_index("id_site")["activity_type"].map(activity_group)

    parts = []
    for ch in pd.read_csv(raw / "studentVle.csv",
                          usecols=["code_module", "code_presentation", "id_student", "id_site", "date", "sum_click"],
                          chunksize=chunksize):
        ch = ch[ch.code_module == module]
        if ch.empty:
            continue
        ch = ch.assign(grp=ch.id_site.map(site_grp)).dropna(subset=["grp"])
        ch = ch.assign(week=np.floor(ch.date / 7).astype(int), grp=ch.grp.astype(int))
        parts.append(ch.groupby(["code_presentation", "id_student", "week", "grp"], as_index=False).sum_click.sum())
    keys = ["code_presentation", "id_student", "week", "grp"]
    if not parts:
        raise ValueError(f"No studentVle click rows found for module {module!r}; check the course code and raw data.")
    weekly = pd.concat(parts, ignore_index=True).groupby(keys, as_index=False).sum_click.sum()

    rows = sa.merge(ass, on="id_assessment")
    rows["week_deadline"] = np.floor(rows.date / 7).astype(int)
    out = []
    for (pres, aid), grp_rows in rows.groupby(["code_presentation", "id_assessment"]):
        wd = int(grp_rows.week_deadline.iloc[0])
        w = weekly[(weekly.code_presentation == pres) & (weekly.week >= wd - 4) & (weekly.week <= wd - 1)]
        piv = w.pivot_table(index="id_student", columns="grp", values="sum_click", aggfunc="sum", fill_value=0)
        piv = piv.reindex(columns=[0, 1, 2], fill_value=0)
        piv.columns = ["clicks_g1", "clicks_g2", "clicks_g3"]
        last = w[w.week == wd - 1].groupby("id_student").sum_click.sum().rename("clicks_last_week")
        g = grp_rows.merge(piv, left_on="id_student", right_index=True, how="left")
        g = g.merge(last, left_on="id_student", right_index=True, how="left")
        out.append(g)
    if not out:
        raise ValueError(f"No scored, dated TMA/CMA assessments found for module {module!r}.")
    df = pd.concat(out, ignore_index=True)
    for c in ["clicks_g1", "clicks_g2", "clicks_g3", "clicks_last_week"]:
        df[c] = df[c].fillna(0)
    df = df.sort_values(["code_presentation", "id_student", "date", "id_assessment"]).reset_index(drop=True)
    gb = df.groupby(["code_presentation", "id_student"])
    df["prev_mean"] = gb.score.transform(lambda s: s.expanding().mean().shift())
    df["prev_last"] = gb.score.shift()
    df["n_prev"] = gb.cumcount()
    df["has_prev"] = (df.n_prev > 0).astype(int)
    df[["prev_mean", "prev_last"]] = df[["prev_mean", "prev_last"]].fillna(50.0)   # neutral constant, has_prev flags it
    df["is_tma"] = (df.assessment_type == "TMA").astype(int)
    df["is_cma"] = (df.assessment_type == "CMA").astype(int)
    df["weeks_since_start"] = df.week_deadline
    keep = ["code_module", "code_presentation", "id_student", "id_assessment", "assessment_type", "date",
            "week_deadline"] + FEATURES + ["score"]
    df = df[keep]
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df
