"""
Download and preprocess US macroeconomic data from FRED.

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
                          filtering sections.
    multivariate.csv      Unstandardised quarterly panel of GDP growth, IP growth,
                          and −ΔUNRATE.

Standardization
---------------
Standardization is intentionally *not* applied here.  Call ``standardize()`` on
the returned DataFrame with the method appropriate for your task:

    'full'      z-score over the full sample — introduces look-ahead bias; only
                valid for retrospective tasks (smoothing, in-sample fit).
    'expanding' z-score using only observations up to each point — look-ahead
                free; suitable for pseudo-out-of-sample evaluation.
    'rolling'   z-score over a trailing window — look-ahead free; requires a
                burn-in period of ``window`` quarters.
    'none'      No standardization; returns the panel unchanged.

Example
-------
    raw        = download_raw()
    Y_raw      = build_multivariate(raw)
    Y_std, loc, scale = standardize(Y_raw, method='expanding')
"""

import io
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import requests

# ── output directory ──────────────────────────────────────────────────────────
OUT_DIR = Path(__file__).parent
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── FRED series definitions ───────────────────────────────────────────────────
FRED_BASE = "https://fred.stlouisfed.org/graph/fredgraph.csv?id="

SERIES = {
    "GDPC1":            "Real Gross Domestic Product",
    "GDPPOT":           "Real Potential GDP (CBO estimate)",
    "A939RX0Q048SBEA":  "Real GDP per Capita",
    "INDPRO":           "Industrial Production Index",
    "UNRATE":           "Civilian Unemployment Rate",
}


def _fetch(series_id: str) -> pd.Series:
    url  = FRED_BASE + series_id
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    df = pd.read_csv(
        io.StringIO(resp.text),
        parse_dates=["observation_date"],
        index_col="observation_date",
    )
    s = pd.to_numeric(df.iloc[:, 0], errors="coerce").rename(series_id)
    print(f"  {series_id:<22}  {len(s):>5} obs   "
          f"{s.index[0].date()} → {s.index[-1].date()}")
    return s


def download_raw() -> dict[str, pd.Series]:
    """Fetch all FRED series and return as a dict of pd.Series."""
    print("Downloading from FRED …")
    raw = {sid: _fetch(sid) for sid in SERIES}
    print()
    return raw


def standardize(
    df: pd.DataFrame,
    method: Literal['full', 'expanding', 'rolling', 'none'] = 'full',
    window: int = 36,
    ddof: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Standardize (z-score) a DataFrame column-by-column.

    Parameters
    ----------
    df : pd.DataFrame
        Input panel.  Each column is standardized independently.
    method : {'full', 'expanding', 'rolling', 'none'}
        'full'      Subtract full-sample mean, divide by full-sample std.
                    **Introduces look-ahead bias** — only valid for smoothing
                    or other purely retrospective tasks.
        'expanding' At each time step t, subtract and divide by the mean/std
                    of rows 0..t.  Look-ahead free; the first row is dropped
                    because std is undefined for a single observation.
        'rolling'   Subtract and divide by the trailing ``window``-period
                    mean/std.  Look-ahead free; the first ``window - 1`` rows
                    are dropped.
        'none'      No standardization.  Returns df unchanged; loc = 0,
                    scale = 1 for every column and time step.
    window : int
        Trailing window length for method='rolling'.  Ignored otherwise.
    ddof : int
        Delta degrees of freedom for std.  Default 1 (unbiased estimator).

    Returns
    -------
    standardized : pd.DataFrame
        Standardized values.  Same columns as ``df``; may have fewer rows for
        'expanding' (drops first) and 'rolling' (drops first window-1).
    loc : pd.DataFrame
        Location (mean) used for each column at each time step.  Same shape
        as ``standardized``.  Constant rows for method='full'.
    scale : pd.DataFrame
        Scale (std) used for each column at each time step.  Same shape as
        ``standardized``.  Constant rows for method='full'.

    Notes
    -----
    ``loc`` and ``scale`` are useful for:
    * Diagnostics: verify that location/scale are strictly backward-looking.
    * Inverse transform: ``original ≈ standardized * scale + loc``.
    * Comparing standardization methods side by side.
    """
    if method == 'none':
        loc   = pd.DataFrame(0.0, index=df.index, columns=df.columns)
        scale = pd.DataFrame(1.0, index=df.index, columns=df.columns)
        return df.copy(), loc, scale

    if method == 'full':
        mu  = df.mean()          # Series — one value per column
        sig = df.std(ddof=ddof)
        loc   = pd.DataFrame([mu]   * len(df), index=df.index, columns=df.columns)
        scale = pd.DataFrame([sig]  * len(df), index=df.index, columns=df.columns)

    elif method == 'expanding':
        loc   = df.expanding(min_periods=2).mean()
        scale = df.expanding(min_periods=2).std(ddof=ddof)
        # Shift so that the standardization at time t uses data up to t-1
        # (otherwise E[x_t] at step t still includes x_t itself).
        loc   = loc.shift(1)
        scale = scale.shift(1)

    elif method == 'rolling':
        loc   = df.rolling(window=window, min_periods=window).mean().shift(1)
        scale = df.rolling(window=window, min_periods=window).std(ddof=ddof).shift(1)

    else:
        raise ValueError(
            f"method={method!r} not recognised.  "
            "Choose 'full', 'expanding', 'rolling', or 'none'."
        )

    standardized = (df - loc) / scale
    # Drop rows where loc or scale is NaN (burn-in for expanding/rolling).
    mask = standardized.notna().all(axis=1)
    return standardized[mask], loc[mask], scale[mask]


def build_univariate(raw: dict[str, pd.Series]) -> pd.DataFrame:
    """
    Quarterly univariate panel.

    Columns
    -------
    gdp_growth   100 × Δ log GDPC1  (quarterly %)
    output_gap   100 × (GDPC1 − GDPPOT) / GDPPOT  (%)
    gdp_pc       Real GDP per capita (index level, 2017 = 100)

    No standardization applied.
    """
    gdp    = raw["GDPC1"]
    gdppot = raw["GDPPOT"]

    gdp_growth = (100.0 * np.log(gdp).diff()).rename("gdp_growth")
    output_gap = (100.0 * (gdp - gdppot) / gdppot).rename("output_gap")

    return pd.concat(
        [gdp_growth, output_gap, raw["A939RX0Q048SBEA"].rename("gdp_pc")],
        axis=1,
    ).dropna()


def build_multivariate(raw: dict[str, pd.Series]) -> pd.DataFrame:
    """
    Unstandardized quarterly panel for the multivariate factor model.

    Columns
    -------
    GDP_growth   100 × Δ log GDPC1  (quarterly %)
    IP_growth    100 × Δ log INDPRO  (quarter-end to quarter-end, %)
    neg_UR_diff  −ΔUNRATE  (quarter-end, sign-flipped → procyclical)

    No standardization applied.  Call ``standardize(df, method=...)`` on the
    result to z-score the panel for use with the factor model.
    """
    gdp_growth = (100.0 * np.log(raw["GDPC1"]).diff()).rename("GDP_growth")

    ip_q      = raw["INDPRO"].resample("QS").last()
    ip_growth = (100.0 * np.log(ip_q).diff()).rename("IP_growth")

    ur_q   = raw["UNRATE"].resample("QS").last()
    neg_ur = (-ur_q.diff()).rename("neg_UR_diff")

    return pd.concat(
        [gdp_growth, ip_growth, neg_ur], axis=1, join="inner"
    ).dropna()


def build_raw_panel(raw: dict[str, pd.Series]) -> pd.DataFrame:
    """
    Level series on a common quarterly (quarter-start) index.
    Monthly series represented by their quarter-end value.
    """
    return pd.concat(
        {
            "GDPC1":           raw["GDPC1"],
            "GDPPOT":          raw["GDPPOT"],
            "A939RX0Q048SBEA": raw["A939RX0Q048SBEA"],
            "INDPRO":          raw["INDPRO"].resample("QS").last(),
            "UNRATE":          raw["UNRATE"].resample("QS").last(),
        },
        axis=1,
        join="outer",
    )


def main():
    raw = download_raw()

    panel_raw = build_raw_panel(raw)
    panel_uni = build_univariate(raw)
    panel_mv  = build_multivariate(raw)

    panel_raw.to_csv(OUT_DIR / "fred_raw.csv")
    panel_uni.to_csv(OUT_DIR / "univariate.csv")
    panel_mv.to_csv(OUT_DIR  / "multivariate.csv")

    print("Saved files:")
    for name, df in [
        ("fred_raw.csv",     panel_raw),
        ("univariate.csv",   panel_uni),
        ("multivariate.csv", panel_mv),
    ]:
        print(f"  {name:<22}  {df.shape[0]} rows × {df.shape[1]} cols   "
              f"{df.index[0].date()} → {df.index[-1].date()}")

    # Show the effect of each standardization method on the multivariate panel.
    print()
    print("Multivariate panel — raw summary:")
    print(panel_mv.describe().round(3).to_string())

    print()
    print("Correlations (raw):")
    print(panel_mv.corr().round(3).to_string())

    for method in ('full', 'expanding', 'rolling'):
        std_df, _, _ = standardize(panel_mv, method=method)
        print(f"\nStandardize method='{method}'  →  {len(std_df)} rows:")
        print(std_df.describe().round(3).to_string())


if __name__ == "__main__":
    main()
