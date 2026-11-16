"""Reproducible data script (fixed seed).  Run from the project root:
  python -m data.make_data --download --module FFF          # real OULAD (needs internet)
  python -m data.make_data --raw-dir data/raw --module FFF  # OULAD CSVs already in data/raw
  python -m data.make_data --synthetic                      # offline dry run with FAKE OULAD-schema data"""
import argparse, shutil, urllib.request, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from config import SEED
from data.preprocessing import build_processed

OULAD_URLS = [
    "https://api.figshare.com/v2/articles/5081998/download",  # public Figshare record
    "https://analyse.kmi.open.ac.uk/open_dataset/download",   # original OU endpoint fallback
]
REQUIRED_TABLES = ("assessments.csv", "studentAssessment.csv", "studentVle.csv", "vle.csv")


def download_oulad(raw_dir: Path) -> None:
    """Download the public OULAD archive, trying Figshare then the OU endpoint.

    The public Figshare API does not require authentication. Some copies of the archive
    place the CSVs in a nested folder, so the required tables are normalized to raw_dir.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    zpath = raw_dir / "oulad.zip"
    errors = []
    for url in OULAD_URLS:
        try:
            print(f"Downloading OULAD from {url} ...")
            request = urllib.request.Request(url, headers={"User-Agent": "StudyPlanner/1.0"})
            with urllib.request.urlopen(request, timeout=90) as response, zpath.open("wb") as target:
                shutil.copyfileobj(response, target)
            if not zipfile.is_zipfile(zpath):
                raise ValueError("The download response was not a ZIP archive.")
            break
        except Exception as exc:
            errors.append(f"{url}: {exc}")
            if zpath.exists():
                zpath.unlink()
    else:
        raise RuntimeError("Could not download OULAD from the available public endpoints:\n" + "\n".join(errors))

    with zipfile.ZipFile(zpath) as archive:
        archive.extractall(raw_dir)
    for table in REQUIRED_TABLES:
        matches = list(raw_dir.rglob(table))
        if not matches:
            raise FileNotFoundError(f"Downloaded OULAD archive does not contain {table}.")
        source = matches[0]
        destination = raw_dir / table
        if source.resolve() != destination.resolve():
            shutil.copy2(source, destination)
    print("OULAD tables ready in", raw_dir)


def make_synthetic_raw(raw_dir, seed: int = SEED, n_students: int = 600, module: str = "SYN") -> None:
    """Writes tables with the SAME schema as OULAD so the same pipeline runs. For offline dry-runs ONLY."""
    rng = np.random.default_rng(seed)
    raw = Path(raw_dir); raw.mkdir(parents=True, exist_ok=True)
    pres = ["2013B", "2013J", "2014B", "2014J"]; n_weeks = 36
    sites = pd.DataFrame({"id_site": range(1, 10), "code_module": module,
                          "activity_type": ["resource", "oucontent", "url", "forumng", "oucollaborate", "ouwiki",
                                            "quiz", "homepage", "externalquiz"]})
    site_ids = {0: [1, 2, 3], 1: [4, 5, 6], 2: [7, 8, 9]}
    dates = [25, 53, 88, 116, 165, 200]; weights = [10, 15, 20, 20, 15, 20]
    types = ["TMA", "CMA", "TMA", "CMA", "TMA", "TMA"]
    a_rows, sa_rows, v_rows = [], [], []
    gm = np.array([1.0, 0.35, 0.6])
    for pi, p in enumerate(pres):
        n = n_students
        ability = rng.normal(size=n)
        eng = 0.6 * ability + 0.8 * rng.normal(size=n)
        base = 24 * np.exp(0.45 * eng)
        weekly = rng.poisson(base[:, None, None] * gm[None, None, :] * np.ones((1, n_weeks, 1)))
        dropout = rng.integers(8, 70, n)
        weekly = weekly * (np.arange(n_weeks)[None, :, None] < dropout[:, None, None])
        ids = np.arange(n) + 1 + pi * n
        s, w, g = np.nonzero(weekly)
        site = np.array([rng.choice(site_ids[x]) for x in g])
        v_rows.append(pd.DataFrame({"code_module": module, "code_presentation": p, "id_student": ids[s],
                                    "id_site": site, "date": w * 7 + rng.integers(0, 7, len(w)),
                                    "sum_click": weekly[s, w, g]}))
        for k in range(6):
            aid = 1000 * (pi + 1) + k
            a_rows.append((module, p, aid, types[k], dates[k], weights[k]))
            wd = dates[k] // 7
            win = weekly[:, max(0, wd - 4):wd, :].sum(axis=(1, 2))
            score = 52 + 11 * ability + 6.0 * np.log1p(win / 40.0) - 8 + rng.normal(0, 7, n)
            sub = (wd < dropout) & (rng.random(n) < 0.92)
            sa_rows.append(pd.DataFrame({"id_assessment": aid, "id_student": ids[sub],
                                         "date_submitted": dates[k] - 1, "is_banked": 0,
                                         "score": np.clip(np.round(score[sub]), 0, 100)}))
        a_rows.append((module, p, 1000 * (pi + 1) + 9, "Exam", np.nan, 100))
    pd.DataFrame(a_rows, columns=["code_module", "code_presentation", "id_assessment", "assessment_type", "date",
                                  "weight"]).to_csv(raw / "assessments.csv", index=False)
    pd.concat(sa_rows).to_csv(raw / "studentAssessment.csv", index=False)
    pd.concat(v_rows).to_csv(raw / "studentVle.csv", index=False)
    sites.to_csv(raw / "vle.csv", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--module", default="FFF", help="OULAD course code (e.g. FFF)")
    ap.add_argument("--out", default="data/processed/oulad_processed.csv")
    ap.add_argument("--seed", type=int, default=SEED)
    a = ap.parse_args()
    raw = Path(a.raw_dir)
    if a.synthetic:
        raw = Path("data/raw_synthetic"); a.module = "SYN"
        make_synthetic_raw(raw, seed=a.seed)
        print("WARNING: synthetic data (schema = OULAD). Use only for dry runs, never for the final report.")
    elif a.download:
        download_oulad(raw)
    df = build_processed(raw, a.module, a.out)
    print(f"Wrote {a.out}: {len(df)} rows, {df.id_student.nunique()} students, presentations {sorted(df.code_presentation.unique())}")


if __name__ == "__main__":
    main()
