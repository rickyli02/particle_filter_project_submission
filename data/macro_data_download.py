"""
Download and preprocess US macroeconomic data from FRED for MC_Projectv3.

Uses FRED's public CSV endpoint — no API key required. Run this script once to
cache data locally; the notebook can then load from the saved CSV files instead
of fetching from the internet on every run.

Usage
-----
    python data/macro_data_download.py

Output files (written to data/)
--------------------------------
    fred_raw.csv          Raw quarterly series (GDPC1, GDPPOT, A939RX0Q048SBEA)
                          and raw monthly series (INDPRO, UNRATE) aligned to a
                          common quarterly index.
    univariate.csv        GDP growth and output gap used in the univariate
                          filtering sections (§3, §3b, §5).
    multivariate.csv      Standardised quarterly panel of GDP growth, IP growth,
                          and −ΔUNRATE used in the multivariate section (§4).
"""

import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# ── output directory ──────────────────────────────────────────────────────────
OUT_DIR = Path(__file__).parent
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── FRED series definitions ───────────────────────────────────────────────────
# All series are available through FRED's public CSV endpoint with no API key.
# Frequency and units are listed as FRED reports them.
FRED_BASE = "https://fred.stlouisfed.org/graph/fredgraph.csv?id="

SERIES = {
    # Quarterly — billions of chained 2017 dollars, seasonally adjusted annual rate
    "GDPC1":            "Real Gross Domestic Product",
    "GDPPOT":           "Real Potential GDP (CBO estimate)",
    # Quarterly — index 2017=100, seasonally adjusted
    "A939RX0Q048SBEA":  "Real GDP per Capita",
    # Monthly — index 2017=100, seasonally adjusted
    "INDPRO":           "Industrial Production Index",
    # Monthly — percent, seasonally adjusted
    "UNRATE":           "Civilian Unemployment Rate",
}


def _fetch(series_id: str) -> pd.Series:
    url = FRED_BASE + series_id
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    df = pd.read_csv(
        io.StringIO(resp.text),
        parse_dates=["observation_date"],
        index_col="observation_date",
    )
    s = df.iloc[:, 0].rename(series_id)
    s = pd.to_numeric(s, errors="coerce")
    print(f"  {series_id:<22}  {len(s):>5} obs   "
          f"{s.index[0].date()} → {s.index[-1].date()}")
    return s


def download_raw() -> dict[str, pd.Series]:
    print("Downloading from FRED …")
    raw = {}
    for sid in SERIES:
        raw[sid] = _fetch(sid)
    print()
    return raw


def build_univariate(raw: dict[str, pd.Series]) -> pd.DataFrame:
    """
    Quarterly univariate panel used in §3, §3b, §5.

      gdp_growth   100 * Δ log GDPC1  (% change)
      output_gap   100 * (GDPC1 − GDPPOT) / GDPPOT
      gdp_pc       Real GDP per capita (level, index)
    """
    gdp      = raw["GDPC1"]
    gdppot   = raw["GDPPOT"]
    gdppc    = raw["A939RX0Q048SBEA"]

    gdp_growth  = 100.0 * gdp.pct_change().rename("gdp_growth")
    output_gap  = (100.0 * (gdp - gdppot) / gdppot).rename("output_gap")

    df = pd.concat([gdp_growth, output_gap, gdppc.rename("gdp_pc")], axis=1).dropna()
    return df


def build_multivariate(raw: dict[str, pd.Series]) -> pd.DataFrame:
    """
    Standardised quarterly panel used in §4.

      GDP_growth   100 * Δ log GDPC1 (quarterly)
      IP_growth    100 * Δ log INDPRO (quarter-end, % change)
      neg_UR_diff  −ΔUNRATE (quarter-end, sign-flipped → procyclical)

    All three series are standardised to zero mean and unit variance after
    alignment on the inner join, matching the preprocessing in the notebook.
    """
    gdp_growth = 100.0 * raw["GDPC1"].pct_change().rename("GDP_growth")

    ip_q       = raw["INDPRO"].resample("QS").last()
    ip_growth  = (100.0 * ip_q.pct_change()).rename("IP_growth")

    ur_q       = raw["UNRATE"].resample("QS").last()
    neg_ur     = (-ur_q.diff()).rename("neg_UR_diff")

    df = pd.concat([gdp_growth, ip_growth, neg_ur], axis=1, join="inner").dropna()

    # standardise
    df = (df - df.mean()) / df.std()
    return df


def build_raw_panel(raw: dict[str, pd.Series]) -> pd.DataFrame:
    """
    Level series resampled to a common quarterly index (quarter-start).
    Monthly series are represented by their quarter-end value.
    """
    quarterly = {
        "GDPC1":            raw["GDPC1"],
        "GDPPOT":           raw["GDPPOT"],
        "A939RX0Q048SBEA":  raw["A939RX0Q048SBEA"],
        "INDPRO":           raw["INDPRO"].resample("QS").last(),
        "UNRATE":           raw["UNRATE"].resample("QS").last(),
    }
    return pd.concat(quarterly, axis=1, join="outer")


def main():
    raw = download_raw()

    panel_raw  = build_raw_panel(raw)
    panel_uni  = build_univariate(raw)
    panel_mv   = build_multivariate(raw)

    # ── save ──────────────────────────────────────────────────────────────────
    panel_raw.to_csv(OUT_DIR / "fred_raw.csv")
    panel_uni.to_csv(OUT_DIR / "univariate.csv")
    panel_mv.to_csv(OUT_DIR  / "multivariate.csv")

    # ── report ────────────────────────────────────────────────────────────────
    print("Saved files:")
    for name, df in [
        ("fred_raw.csv",     panel_raw),
        ("univariate.csv",   panel_uni),
        ("multivariate.csv", panel_mv),
    ]:
        print(f"  {name:<22}  {df.shape[0]} rows × {df.shape[1]} cols   "
              f"{df.index[0].date()} → {df.index[-1].date()}")

    print()
    print("Multivariate panel summary (standardised):")
    print(panel_mv.describe().round(3).to_string())
    print()
    print("Correlations:")
    print(panel_mv.corr().round(3).to_string())


if __name__ == "__main__":
    main()
