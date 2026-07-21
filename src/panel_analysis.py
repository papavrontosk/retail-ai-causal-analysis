"""
panel_analysis.py
=================
Retail Personalization Engine — Panel Data Causal Analysis
Author : Konstantinos Papavrontos
Purpose: Reusable pipeline that reproduces all results from the assignment.

Usage
-----
    python panel_analysis.py

    # Override default paths:
    DATA_DIR=./my_data OUTPUT_DIR=./my_out python panel_analysis.py

Inputs  (place in ./data/ or set DATA_DIR)
-------
    store_info.csv
    algo_rollout.csv
    store_month_metrics.csv

Outputs (written to ./outputs/ or set OUTPUT_DIR)
-------
    q2_time_trends.png
    q10_ranking.png
    results_summary.csv
"""

# ── Standard library ──────────────────────────────────────────────────────────
import os
import warnings
warnings.filterwarnings("ignore")

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from linearmodels.panel import PanelOLS

# ── Configuration ─────────────────────────────────────────────────────────────
DATA_DIR   = os.environ.get("DATA_DIR",   "./data")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 130,
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# Variables used in every regression
EXOG_VARS = ["ai_share", "web_visits_k", "competitor_price", "ad_spend_k", "stockout_rate"]

FD_X_COLS = [
    "d_ai_share", "d_ai_lag1", "d_ai_lag2", "d_ai_lead1",
    "d_web_visits_k", "d_competitor_price", "d_ad_spend_k", "d_stockout_rate",
]

LD_VARS = ["ln_sales", "ai_share", "web_visits_k", "competitor_price", "ad_spend_k", "stockout_rate"]


# ══════════════════════════════════════════════════════════════════════════════
# 1.  DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_data(data_dir: str = DATA_DIR):
    """Load and parse the three source CSV files."""
    si = pd.read_csv(os.path.join(data_dir, "store_info.csv"))
    ar = pd.read_csv(os.path.join(data_dir, "algo_rollout.csv"))
    me = pd.read_csv(os.path.join(data_dir, "store_month_metrics.csv"))

    si["opening_month"]  = pd.to_datetime(si["opening_month"])
    si["closing_month"]  = pd.to_datetime(si["closing_month"])
    ar["go_live_month"]  = pd.to_datetime(ar["go_live_month"], errors="coerce")
    me["month"]          = pd.to_datetime(me["month"])

    print(f"Loaded  store_info {si.shape}  |  algo_rollout {ar.shape}  |  metrics {me.shape}")
    return si, ar, me


# ══════════════════════════════════════════════════════════════════════════════
# 2.  ANALYSIS SAMPLE  (Question 1)
# ══════════════════════════════════════════════════════════════════════════════

def build_analysis_sample(
    store_info, algo_rollout, metrics,
    min_months: int = 20,
    min_pre: int    = 5,
    min_post: int   = 5,
) -> pd.DataFrame:
    """
    Merge and filter to the main analysis sample.

    Steps
    -----
    1. Merge all three frames on store_id.
    2. Keep store-months within each store's operating window.
    3. Drop months with report_flag == 0.
    4. Keep stores with >= min_months reported months.
    5. For treated stores only: require >= min_pre months before
       and >= min_post months on/after go_live_month.
    """
    df = metrics.merge(store_info, on="store_id").merge(algo_rollout, on="store_id")

    df = df[(df["month"] >= df["opening_month"]) & (df["month"] <= df["closing_month"])]
    print(f"Step 1 – operating window    : {len(df):>5} obs | {df['store_id'].nunique()} stores")

    df = df[df["report_flag"] == 1]
    print(f"Step 2 – report_flag == 1    : {len(df):>5} obs | {df['store_id'].nunique()} stores")

    counts = df.groupby("store_id")["month"].count()
    df = df[df["store_id"].isin(counts[counts >= min_months].index)]
    print(f"Step 3 – >= {min_months} months         : {len(df):>5} obs | {df['store_id'].nunique()} stores")

    def _treated_ok(grp):
        if grp["treated"].iloc[0] == 0:
            return True
        go = grp["go_live_month"].iloc[0]
        if pd.isna(go):
            return False
        return (grp["month"] < go).sum() >= min_pre and (grp["month"] >= go).sum() >= min_post

    df = df.groupby("store_id").filter(_treated_ok).copy()
    print(f"Step 4 – treated balance     : {len(df):>5} obs | {df['store_id'].nunique()} stores")

    # Derived helpers used downstream
    df["month_str"] = df["month"].dt.to_period("M").astype(str)
    df["mn"]        = df["month"].dt.year * 12 + df["month"].dt.month

    return df


def describe_sample(df: pd.DataFrame) -> None:
    """Print a concise summary of the analysis sample."""
    mps = df.groupby("store_id")["month"].count()
    print("\n" + "=" * 48)
    print(f"  Stores in sample      : {df['store_id'].nunique()}")
    print(f"  Total observations    : {len(df)}")
    print(f"  Treated stores        : {df[df['treated']==1]['store_id'].nunique()}")
    print(f"  Never-treated stores  : {df[df['treated']==0]['store_id'].nunique()}")
    print(f"  Months per store      : {mps.min()}–{mps.max()}")
    print(f"  Panel balanced        : {mps.nunique() == 1}")
    print("=" * 48 + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# 3.  STATUS LABELLING & PLOTS  (Question 2)
# ══════════════════════════════════════════════════════════════════════════════

def assign_status(df: pd.DataFrame) -> pd.DataFrame:
    """
    Label each store-month as:
      'Never treated' | 'Not yet live' | 'Live'
    """
    def _status(row):
        if row["treated"] == 0 or pd.isna(row["go_live_month"]):
            return "Never treated"
        return "Live" if row["month"] >= row["go_live_month"] else "Not yet live"

    df = df.copy()
    df["status"] = df.apply(_status, axis=1)
    return df


def plot_time_trends(df: pd.DataFrame, save_path: str = None) -> None:
    """
    Two-panel chart: mean ln_sales and mean ai_share
    by treatment-status group over calendar time.
    """
    COLORS = {"Never treated": "#64748b", "Not yet live": "#d97706", "Live": "#2563eb"}

    avg_ln = df.groupby(["month_str", "status"])["ln_sales"].mean().unstack()
    avg_ai = df.groupby(["month_str", "status"])["ai_share"].mean().unstack()

    months  = avg_ln.index.tolist()
    x       = range(len(months))
    xticks  = [i for i, m in enumerate(months) if m.endswith("-01") or m.endswith("-07")]
    xlabels = [months[i] for i in xticks]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Mean outcomes by treatment-status group", fontsize=13, fontweight="bold")

    for status in ["Never treated", "Not yet live", "Live"]:
        c = COLORS[status]
        if status in avg_ln.columns:
            axes[0].plot(list(x), avg_ln[status].values.astype(float),
                         color=c, lw=2.2, label=status, marker="o", ms=3.5)
        if status in avg_ai.columns:
            axes[1].plot(list(x), avg_ai[status].values.astype(float),
                         color=c, lw=2.2, label=status, marker="o", ms=3.5)

    for ax, title, ylabel in zip(
        axes,
        ["Mean ln(Sales)", "Mean ai_share (%)"],
        ["ln(Net Sales)",  "ai_share (%)"],
    ):
        ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
        ax.set_ylabel(ylabel)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabels, rotation=45, ha="right", fontsize=9)
        ax.legend(fontsize=9, framealpha=0.5)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Chart saved → {save_path}")
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# 4.  FIXED EFFECTS – LEVELS  (Question 3)
# ══════════════════════════════════════════════════════════════════════════════

def run_fe(df: pd.DataFrame, label: str = "FE") -> dict:
    """
    Two-way FE (store + month) with store-clustered SE via linearmodels.

    Returns dict: params, std_errors, pvalues, n_stores, n_obs, label
    """
    idx   = df.set_index(["store_id", "month"])
    model = PanelOLS(idx["ln_sales"], idx[EXOG_VARS],
                     entity_effects=True, time_effects=True)
    res   = model.fit(cov_type="clustered", cluster_entity=True)

    n_stores = idx.index.get_level_values(0).nunique()
    n_obs    = len(idx)

    print(f"\n{'='*55}")
    print(f"  {label}   |   {n_stores} stores, {n_obs} obs")
    print(f"{'='*55}")
    tbl = pd.DataFrame({
        "Coef.":    res.params,
        "Std.Err.": res.std_errors,
        "p-value":  res.pvalues,
    })
    print(tbl.round(6).to_string())
    ai = res.params["ai_share"]
    print(f"\n  → 10pp effect on ln(sales) : {ai*10:.4f}  ≈ {ai*10*100:.2f}%")

    return dict(params=res.params, std_errors=res.std_errors, pvalues=res.pvalues,
                n_stores=n_stores, n_obs=n_obs, label=label, result_obj=res)


# ══════════════════════════════════════════════════════════════════════════════
# 5.  FIRST DIFFERENCES  (Question 4)
# ══════════════════════════════════════════════════════════════════════════════

def build_fd_base(df: pd.DataFrame) -> pd.DataFrame:
    """
    Strict first differences: only when the previous observation is
    exactly the preceding calendar month (no gaps allowed).
    """
    df = df.sort_values(["store_id", "month"]).copy()
    rows = []
    for sid, grp in df.groupby("store_id"):
        grp = grp.sort_values("month").reset_index(drop=True)
        for i in range(1, len(grp)):
            c, p = grp.iloc[i], grp.iloc[i - 1]
            if c["mn"] - p["mn"] != 1:       # gap detected → skip
                continue
            rows.append({
                "store_id":           sid,
                "month":              c["month"],
                "mn":                 c["mn"],
                "d_ln_sales":         c["ln_sales"]         - p["ln_sales"],
                "d_ai_share":         c["ai_share"]         - p["ai_share"],
                "d_web_visits_k":     c["web_visits_k"]     - p["web_visits_k"],
                "d_competitor_price": c["competitor_price"] - p["competitor_price"],
                "d_ad_spend_k":       c["ad_spend_k"]       - p["ad_spend_k"],
                "d_stockout_rate":    c["stockout_rate"]    - p["stockout_rate"],
            })
    return pd.DataFrame(rows)


def add_ai_lags(fd: pd.DataFrame) -> pd.DataFrame:
    """
    Add Δai_share_{t-1}, Δai_share_{t-2}, Δai_share_{t+1} to the FD frame.
    Each lag/lead is also computed only over strictly consecutive FD rows.
    Rows with any missing lag/lead are dropped.
    """
    out = []
    for sid, grp in fd.groupby("store_id"):
        grp = grp.sort_values("month").reset_index(drop=True)
        mn  = grp["mn"].values
        da  = grp["d_ai_share"].values
        grp["d_ai_lag1"]  = np.nan
        grp["d_ai_lag2"]  = np.nan
        grp["d_ai_lead1"] = np.nan
        for i in range(len(grp)):
            if i > 0 and mn[i] - mn[i-1] == 1:
                grp.at[i, "d_ai_lag1"] = da[i-1]
            if i > 1 and mn[i]-mn[i-1] == 1 and mn[i-1]-mn[i-2] == 1:
                grp.at[i, "d_ai_lag2"] = da[i-2]
            if i < len(grp)-1 and mn[i+1]-mn[i] == 1:
                grp.at[i, "d_ai_lead1"] = da[i+1]
        out.append(grp)
    return pd.concat(out, ignore_index=True).dropna(
        subset=["d_ai_lag1", "d_ai_lag2", "d_ai_lead1"]
    )


def build_fd_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Full FD pipeline: base differences → lags/leads → month_str helper."""
    base = build_fd_base(df)
    fd   = add_ai_lags(base)
    fd["month_str"] = fd["month"].dt.to_period("M").astype(str)
    print(f"FD dataset: {len(fd)} rows, {fd['store_id'].nunique()} stores")
    return fd


def _cumulative_se(model, v0="d_ai_share", v1="d_ai_lag1", v2="d_ai_lag2") -> float:
    """SE of (β₀ + β₁ + β₂) via the delta method (variance of linear combination)."""
    cov = model.cov_params()
    var = (
        cov.loc[v0, v0] + cov.loc[v1, v1] + cov.loc[v2, v2]
        + 2*cov.loc[v0, v1] + 2*cov.loc[v0, v2] + 2*cov.loc[v1, v2]
    )
    return float(np.sqrt(var))


def run_fd(
    fd: pd.DataFrame,
    store_trends: bool = False,
    label: str = "FD",
) -> dict:
    """
    Estimate FD regression with month FE and optionally store dummies
    (store-specific linear trends in the levels equation).

    Parameters
    ----------
    fd           : output of build_fd_dataset()
    store_trends : True  → FD-B (month FE + store dummies)
                   False → FD-A (month FE only)
    label        : header label

    Returns
    -------
    dict: params, bse, pvalues, cum_effect, cum_se, n_stores, n_obs, label
    """
    month_dummies = pd.get_dummies(fd["month_str"], prefix="m", drop_first=True)
    X = pd.concat([fd[FD_X_COLS], month_dummies], axis=1)
    if store_trends:
        store_dummies = pd.get_dummies(fd["store_id"], prefix="s", drop_first=True)
        X = pd.concat([X, store_dummies], axis=1)
    X = sm.add_constant(X.astype(float))
    y = fd["d_ln_sales"].astype(float)

    model = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": fd["store_id"]})

    cum    = sum(model.params[v] for v in ["d_ai_share", "d_ai_lag1", "d_ai_lag2"])
    se_cum = _cumulative_se(model)

    n_stores = fd["store_id"].nunique()
    n_obs    = len(fd)

    print(f"\n{'='*55}")
    print(f"  {label}   |   {n_stores} stores, {n_obs} obs")
    print(f"{'='*55}")
    for v in FD_X_COLS:
        stars = ("***" if model.pvalues[v] < 0.01 else
                 "**"  if model.pvalues[v] < 0.05 else
                 "*"   if model.pvalues[v] < 0.10 else "  ")
        print(f"  {v:<28} {model.params[v]:>10.6f}  SE {model.bse[v]:.6f}  "
              f"p={model.pvalues[v]:.4f} {stars}")
    print(f"\n  Cumulative 3-month (β₀+β₁+β₂) : {cum:.6f}  SE {se_cum:.6f}")
    print(f"  → 10pp cumulative effect        : {cum*10*100:.2f}%")

    return dict(params=model.params, bse=model.bse, pvalues=model.pvalues,
                cum_effect=cum, cum_se=se_cum,
                n_stores=n_stores, n_obs=n_obs, label=label, model_obj=model)


# ══════════════════════════════════════════════════════════════════════════════
# 6.  REDUCED-T BALANCED SAMPLE  (Question 5)
# ══════════════════════════════════════════════════════════════════════════════

def build_reduced_t(df: pd.DataFrame, window_len: int = 12):
    """
    Find the consecutive calendar window of length `window_len` months
    that maximises the number of fully observed stores.

    Returns (df_rt, best_window)
    """
    all_months  = sorted(df["month"].unique())
    best_window = None
    best_count  = 0
    best_stores = None

    for i in range(len(all_months) - window_len + 1):
        win  = all_months[i:i + window_len]
        nums = [m.year * 12 + m.month for m in win]
        if nums[-1] - nums[0] != window_len - 1:   # truly consecutive?
            continue
        cnt    = df[df["month"].isin(win)].groupby("store_id")["month"].count()
        n_full = (cnt == window_len).sum()
        if n_full > best_count:
            best_count  = n_full
            best_window = win
            best_stores = cnt[cnt == window_len].index.tolist()

    df_rt = df[df["store_id"].isin(best_stores) & df["month"].isin(best_window)].copy()

    print(f"\nReduced-T window  : {best_window[0].strftime('%Y-%m')} → {best_window[-1].strftime('%Y-%m')}")
    print(f"Stores            : {df_rt['store_id'].nunique()}")
    print(f"Observations      : {len(df_rt)}")
    print(f"  Treated         : {df_rt[df_rt['treated']==1]['store_id'].nunique()}")
    print(f"  Never-treated   : {df_rt[df_rt['treated']==0]['store_id'].nunique()}")
    print(f"Panel balanced    : {df_rt.groupby('store_id')['month'].count().nunique() == 1}")

    return df_rt, best_window


# ══════════════════════════════════════════════════════════════════════════════
# 7.  REDUCED-N SAMPLE  (Question 7)
# ══════════════════════════════════════════════════════════════════════════════

def build_reduced_n(df_main, store_info, metrics, max_missing: int = 4) -> pd.DataFrame:
    """
    From the main analysis sample keep only stores with
    missing_reports <= max_missing within their operating window.

    missing_reports is counted from the raw metrics frame (before the
    report_flag filter), so it captures months when the store simply
    failed to report.
    """
    raw = metrics.merge(store_info, on="store_id")
    raw = raw[(raw["month"] >= raw["opening_month"]) & (raw["month"] <= raw["closing_month"])]
    missing_ps = (
        raw[raw["report_flag"] == 0]
        .groupby("store_id").size()
        .rename("missing_reports")
    )

    mr       = missing_ps.reindex(df_main["store_id"].unique(), fill_value=0)
    df_rn    = df_main[df_main["store_id"].isin(mr[mr <= max_missing].index)].copy()

    print(f"\nReduced-N (missing <= {max_missing})")
    print(f"  Stores        : {df_rn['store_id'].nunique()}")
    print(f"  Observations  : {len(df_rn)}")
    print(f"  Treated       : {df_rn[df_rn['treated']==1]['store_id'].nunique()}")
    print(f"  Never-treated : {df_rn[df_rn['treated']==0]['store_id'].nunique()}")
    return df_rn


# ══════════════════════════════════════════════════════════════════════════════
# 8.  LONG DIFFERENCE  (Question 8)
# ══════════════════════════════════════════════════════════════════════════════

def run_long_difference(df_rt: pd.DataFrame, best_window: list) -> dict:
    """
    OLS on long differences: first vs last month of the reduced-T window.

    Returns dict: params, bse, pvalues, n_stores, n_obs, label
    """
    first_m, last_m = best_window[0], best_window[-1]
    print(f"\nLong difference: {first_m.strftime('%Y-%m')}  →  {last_m.strftime('%Y-%m')}")

    df_f = df_rt[df_rt["month"] == first_m][["store_id"] + LD_VARS]
    df_l = df_rt[df_rt["month"] == last_m ][["store_id"] + LD_VARS]

    ld = df_f.merge(df_l, on="store_id", suffixes=("_f", "_l"))
    for v in LD_VARS:
        ld[f"ld_{v}"] = ld[f"{v}_l"] - ld[f"{v}_f"]

    LD_X = [f"ld_{v}" for v in LD_VARS if v != "ln_sales"]
    X_ld = sm.add_constant(ld[LD_X])
    y_ld = ld["ld_ln_sales"]
    m_ld = sm.OLS(y_ld, X_ld).fit()

    print(f"Stores in regression: {len(ld)}")
    print(m_ld.summary2().tables[1][["Coef.", "Std.Err.", "t", "P>|t|"]].round(6).to_string())
    ai = m_ld.params["ld_ai_share"]
    print(f"\n  → 10pp effect : {ai*10:.4f}  ≈ {ai*10*100:.2f}%")

    return dict(params=m_ld.params, bse=m_ld.bse, pvalues=m_ld.pvalues,
                n_stores=len(ld), n_obs=len(ld),
                label="Long Diff – Reduced-T", model_obj=m_ld)


# ══════════════════════════════════════════════════════════════════════════════
# 9.  COMPARISON TABLE & RANKING CHART  (Question 10)
# ══════════════════════════════════════════════════════════════════════════════

def build_comparison_table(ranked: list) -> pd.DataFrame:
    """Build a formatted comparison DataFrame from a list of ranked result dicts."""
    rows = []
    for r in ranked:
        rows.append({
            "Rank":       r["rank"],
            "Estimator":  r["label"],
            "10pp Effect":f"{r['effect_10pp']*100:.2f}%",
            "Stores":     r["n_stores"],
            "Obs":        r["n_obs"],
            "Strength":   r["strength"],
            "Weakness":   r["weakness"],
        })
    return pd.DataFrame(rows)


def plot_ranking(ranked: list, save_path: str = None) -> None:
    """Horizontal bar chart of all candidate estimates, coloured by rank."""
    PALETTE = ["#1d4ed8","#2563eb","#3b82f6","#6366f1",
               "#8b5cf6","#a78bfa","#c4b5fd","#e9d5ff"]

    labels  = [r["label"]           for r in ranked]
    effects = [r["effect_10pp"]*100 for r in ranked]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(labels[::-1], effects[::-1],
                   color=PALETTE[::-1], edgecolor="white", height=0.6)
    ax.set_xlabel("Effect of 10pp increase in ai_share on ln(sales) (%)", fontsize=10)
    ax.set_title("Candidate estimates — #1 most reliable to #8 least reliable",
                 fontweight="bold", fontsize=12)
    ax.axvline(0, color="black", lw=0.8, ls="--")
    for bar, val, r in zip(bars, effects[::-1], list(reversed(ranked))):
        ax.text(bar.get_width() + 0.03,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}%  [#{r['rank']}]",
                va="center", fontsize=9)
    ax.set_xlim(-0.5, max(effects) + 0.9)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Chart saved → {save_path}")
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# 10. MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "▓"*60)
    print("  Retail Personalization Engine — Panel Data Analysis")
    print("▓"*60 + "\n")

    # ── Load raw data ─────────────────────────────────────────────────────────
    store_info, algo_rollout, metrics = load_data()

    # ── Q1: Build main analysis sample ────────────────────────────────────────
    print("\n── Q1: Building analysis sample ─────────────────────────")
    df = build_analysis_sample(store_info, algo_rollout, metrics)
    describe_sample(df)

    # ── Q2: Time-trend charts ─────────────────────────────────────────────────
    print("\n── Q2: Time-trend charts ────────────────────────────────")
    df = assign_status(df)
    plot_time_trends(df, save_path=os.path.join(OUTPUT_DIR, "q2_time_trends.png"))

    # ── Q3: Two-way FE on main sample ─────────────────────────────────────────
    print("\n── Q3: Two-way FE – main sample ─────────────────────────")
    fe_main = run_fe(df, label="FE – Main")

    # ── Q4: FD-A and FD-B on main sample ─────────────────────────────────────
    print("\n── Q4: First-difference regressions ─────────────────────")
    fd_main  = build_fd_dataset(df)
    fda_main = run_fd(fd_main, store_trends=False, label="FD-A – Main (month FE only)")
    fdb_main = run_fd(fd_main, store_trends=True,  label="FD-B – Main (month FE + store trends)")

    # ── Q5: Reduced-T balanced sample ─────────────────────────────────────────
    print("\n── Q5: Reduced-T balanced sample ────────────────────────")
    df_rt, best_window = build_reduced_t(df)

    # ── Q6: FE and FD-B on Reduced-T ─────────────────────────────────────────
    print("\n── Q6: Estimations on Reduced-T ─────────────────────────")
    fe_rt  = run_fe(df_rt, label="FE – Reduced-T")
    fd_rt  = build_fd_dataset(df_rt)
    fdb_rt = run_fd(fd_rt, store_trends=True, label="FD-B – Reduced-T")

    # ── Q7: Reduced-N sample ──────────────────────────────────────────────────
    print("\n── Q7: Reduced-N sample ─────────────────────────────────")
    df_rn  = build_reduced_n(df, store_info, metrics, max_missing=4)
    fe_rn  = run_fe(df_rn, label="FE – Reduced-N")
    fd_rn  = build_fd_dataset(df_rn)
    fdb_rn = run_fd(fd_rn, store_trends=True, label="FD-B – Reduced-N")

    # ── Q8: Long difference ────────────────────────────────────────────────────
    print("\n── Q8: Long difference – Reduced-T ─────────────────────")
    ld_rt = run_long_difference(df_rt, best_window)

    # ── Q10: Ranked comparison table & chart ──────────────────────────────────
    print("\n── Q10: Comparative table & ranking ─────────────────────")
    ranked = [
        dict(rank=1, label="FD-B – Main",       n_stores=fdb_main["n_stores"], n_obs=fdb_main["n_obs"],
             effect_10pp=fdb_main["cum_effect"]*10,
             strength="Pre-trend passed; absorbs store trends; largest FD sample",
             weakness="Dynamic controls reduce power; loses 4 stores vs FE"),
        dict(rank=2, label="FD-A – Main",        n_stores=fda_main["n_stores"], n_obs=fda_main["n_obs"],
             effect_10pp=fda_main["cum_effect"]*10,
             strength="Pre-trend passed; controls common time shocks; large sample",
             weakness="Does not absorb store-specific linear trends"),
        dict(rank=3, label="FE – Main",           n_stores=fe_main["n_stores"],  n_obs=fe_main["n_obs"],
             effect_10pp=fe_main["params"]["ai_share"]*10,
             strength="Largest sample; removes time-invariant heterogeneity; most efficient",
             weakness="No pre-trend test; cannot control time-varying confounders"),
        dict(rank=4, label="FD-B – Reduced-N",   n_stores=fdb_rn["n_stores"],  n_obs=fdb_rn["n_obs"],
             effect_10pp=fdb_rn["cum_effect"]*10,
             strength="Clean reporting data; consistent with main; lead negative",
             weakness="Small sample; may select specific store types"),
        dict(rank=5, label="FE – Reduced-N",      n_stores=fe_rn["n_stores"],   n_obs=fe_rn["n_obs"],
             effect_10pp=fe_rn["params"]["ai_share"]*10,
             strength="Cleaner data than main FE; consistent coefficient",
             weakness="Smaller N; potential selection of low-missingness stores"),
        dict(rank=6, label="Long Diff – Red-T",  n_stores=ld_rt["n_stores"],   n_obs=ld_rt["n_obs"],
             effect_10pp=ld_rt["params"]["ld_ai_share"]*10,
             strength="Sweeps intermediate noise; transparent design",
             weakness="Cross-sectional; tiny N; no time controls; no pre-trend test"),
        dict(rank=7, label="FE – Reduced-T",      n_stores=fe_rt["n_stores"],   n_obs=fe_rt["n_obs"],
             effect_10pp=fe_rt["params"]["ai_share"]*10,
             strength="Balanced panel; consistent point estimate",
             weakness="Only 12 stores; very wide CIs; arbitrary window"),
        dict(rank=8, label="FD-B – Reduced-T",   n_stores=fdb_rt["n_stores"],  n_obs=fdb_rt["n_obs"],
             effect_10pp=fdb_rt["cum_effect"]*10,
             strength="Most restrictive specification; balanced design",
             weakness="Too few obs; completely imprecise; sign instability"),
    ]

    comp = build_comparison_table(ranked)
    print("\n" + comp[["Rank","Estimator","10pp Effect","Stores","Obs"]].to_string(index=False))

    out_csv = os.path.join(OUTPUT_DIR, "results_summary.csv")
    comp.to_csv(out_csv, index=False)
    print(f"\nResults table saved → {out_csv}")

    plot_ranking(ranked, save_path=os.path.join(OUTPUT_DIR, "q10_ranking.png"))

    print("\n" + "▓"*60)
    print("  Done. Outputs written to:", os.path.abspath(OUTPUT_DIR))
    print("▓"*60 + "\n")


if __name__ == "__main__":
    main()
