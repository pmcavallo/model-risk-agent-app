"""
Layer 1 — Synthetic model-risk monitoring data.

This script writes four files into data/ that the rest of the demo reads:

    model_registry.csv   which models exist and what type they are
    pd_metrics.csv       12 months of monitoring metrics for the 2 PD models
    lgd_metrics.csv      12 months of monitoring metrics for the 2 LGD models
    policy_rules.json    the thresholds an analyst compares those metrics against

The data is DESIGNED to tell a story so the agent has something real to reason about:

    MDL-A  retail_pd_logit      PD   stable and healthy
    MDL-B  smb_pd_logit         PD   an OBVIOUS breach (PSI blows past 0.25)
    MDL-C  secured_lgd_frac     LGD  stable and healthy
    MDL-D  unsecured_lgd_frac   LGD  the tricky one: every metric drifts one
                                     direction for 12 months but NONE breaches
                                     its threshold. A script says "fine." The
                                     agent, reading the trajectory, flags it.

Everything is synthetic. No real bank, client, or proprietary data is used.

Run it with:  python src/generate_data.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

# A fixed seed makes the output identical every run, so the story never changes.
SEED = 42

# The 12 monthly periods, 2025-01 through 2025-12.
PERIODS = [f"2025-{month:02d}" for month in range(1, 13)]

# Where the files go. This resolves to the project's data/ folder regardless of
# where you run the script from.
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def build_registry():
    """The list of models under monitoring. One row per model."""
    rows = [
        # model_id, model_name,          model_type, method,               deployed
        ("MDL-A", "retail_pd_logit",    "PD",  "simple logistic",     "2024-12"),
        ("MDL-B", "smb_pd_logit",       "PD",  "simple logistic",     "2024-12"),
        ("MDL-C", "secured_lgd_frac",   "LGD", "fractional logistic", "2024-12"),
        ("MDL-D", "unsecured_lgd_frac", "LGD", "fractional logistic", "2024-12"),
    ]
    return pd.DataFrame(
        rows,
        columns=["model_id", "model_name", "model_type", "method", "deployed"],
    )


def build_pd_metrics(rng):
    """
    Monitoring metrics for the 2 PD (probability-of-default) models.

    PD models are judged on discrimination (AUC, KS) and drift (PSI).

    We start from the intended value for each month, then add a little random
    noise so the numbers look real instead of hand-typed.
    """
    rows = []

    # ---- MDL-A: stable and healthy ----------------------------------------
    # AUC and KS hold flat; PSI creeps up but never even reaches 0.10.
    a_psi = np.linspace(0.03, 0.08, 12)  # 0.03 -> 0.08, evenly spaced
    for i, period in enumerate(PERIODS):
        rows.append({
            "model_id": "MDL-A",
            "period": period,
            "auc": 0.780 + rng.normal(0, 0.004),
            "ks":  0.420 + rng.normal(0, 0.004),
            "psi": a_psi[i] + rng.normal(0, 0.004),
            "default_rate_actual":   3.0 + rng.normal(0, 0.1),
            "default_rate_expected": 3.0,
            "volume": int(20000 + rng.normal(0, 300)),
        })

    # ---- MDL-B: an obvious breach -----------------------------------------
    # These exact monthly sequences are the designed story: PSI crosses 0.25 at
    # 2025-08 (the 8th month), and AUC falls below the 0.70 floor around month 10.
    b_psi = [0.04, 0.06, 0.07, 0.09, 0.12, 0.16, 0.20, 0.27, 0.31, 0.35, 0.38, 0.41]
    b_auc = [0.752, 0.749, 0.746, 0.742, 0.735, 0.727, 0.716, 0.703, 0.692, 0.681, 0.671, 0.661]
    b_ks  = np.linspace(0.381, 0.271, 12)          # discrimination decays
    b_dra = np.linspace(3.5, 6.5, 12)              # actual default rate climbs
    for i, period in enumerate(PERIODS):
        rows.append({
            "model_id": "MDL-B",
            "period": period,
            # PSI and AUC are the load-bearing numbers, so keep them exact.
            "auc": b_auc[i],
            "ks":  b_ks[i] + rng.normal(0, 0.004),
            "psi": b_psi[i],
            "default_rate_actual":   b_dra[i] + rng.normal(0, 0.1),
            "default_rate_expected": 3.6,
            "volume": int(15000 + rng.normal(0, 300)),
        })

    frame = pd.DataFrame(rows)

    # Round to the precision an analyst would actually report.
    for col in ["auc", "ks", "psi"]:
        frame[col] = frame[col].round(3)
    for col in ["default_rate_actual", "default_rate_expected"]:
        frame[col] = frame[col].round(2)
    frame["volume"] = frame["volume"].astype(int)
    return frame


def build_lgd_metrics(rng):
    """
    Monitoring metrics for the 2 LGD (loss-given-default) models.

    LGD models are judged on fit (pseudo-R2, RMSE), calibration
    (calibration_error), and drift (PSI). AUC and KS do NOT apply to LGD.
    """
    rows = []

    # ---- MDL-C: stable and healthy ----------------------------------------
    c_psi = np.linspace(0.03, 0.08, 12)
    for i, period in enumerate(PERIODS):
        rows.append({
            "model_id": "MDL-C",
            "period": period,
            "pseudo_r2": 0.340 + rng.normal(0, 0.004),
            "rmse":      0.120 + rng.normal(0, 0.002),
            "calibration_error": 0.010 + rng.normal(0, 0.002),
            "psi": c_psi[i] + rng.normal(0, 0.004),
            "mean_lgd_actual":    0.40 + rng.normal(0, 0.01),
            "mean_lgd_predicted": 0.40,
            "volume": int(8000 + rng.normal(0, 200)),
        })

    # ---- MDL-D: the tricky one --------------------------------------------
    # Every metric below drifts steadily one direction for a full year, yet each
    # stays on the safe side of its threshold. This monotonic sub-threshold
    # drift is the whole point of the demo, so we use the exact designed
    # sequences with NO noise on the drifting metrics — even a tiny wobble could
    # break the strict month-over-month monotonicity the agent needs to detect.
    d_psi = [0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.20]
    d_cal = [0.010, 0.013, 0.017, 0.021, 0.025, 0.029, 0.033, 0.037, 0.040, 0.043, 0.046, 0.048]
    d_rmse = np.linspace(0.130, 0.180, 12)   # monotonic up, stays < 0.20
    d_r2   = np.linspace(0.300, 0.240, 12)   # monotonic DOWN, stays > 0.20
    d_lgd_actual = np.linspace(0.55, 0.66, 12)  # realized loss creeps up
    for i, period in enumerate(PERIODS):
        rows.append({
            "model_id": "MDL-D",
            "period": period,
            "pseudo_r2": d_r2[i],
            "rmse":      d_rmse[i],
            "calibration_error": d_cal[i],
            "psi": d_psi[i],
            "mean_lgd_actual":    d_lgd_actual[i],
            "mean_lgd_predicted": 0.55,
            "volume": int(9000 + rng.normal(0, 200)),
        })

    frame = pd.DataFrame(rows)

    for col in ["pseudo_r2", "rmse", "calibration_error", "psi"]:
        frame[col] = frame[col].round(3)
    for col in ["mean_lgd_actual", "mean_lgd_predicted"]:
        frame[col] = frame[col].round(2)
    frame["volume"] = frame["volume"].astype(int)
    return frame


def build_policy_rules():
    """The thresholds an analyst compares the metrics against."""
    return {
        "pd": {
            "psi": {
                "ok_max": 0.10, "monitor_max": 0.25,
                "actions": {
                    "ok": "No action",
                    "monitor": "Investigate and document",
                    "breach": "Trigger revalidation",
                },
            },
            "auc": {"floor": 0.70, "material_decline_from_dev": 0.05},
            "ks":  {"floor": 0.30, "material_decline_from_dev": 0.10},
            "default_rate": {"actual_vs_expected_tolerance_pct": 25},
        },
        "lgd": {
            "psi": {
                "ok_max": 0.10, "monitor_max": 0.25,
                "actions": {
                    "ok": "No action",
                    "monitor": "Investigate and document",
                    "breach": "Trigger revalidation",
                },
            },
            "pseudo_r2":         {"floor": 0.20, "material_decline_from_dev": 0.05},
            "rmse":              {"ceiling": 0.20, "material_increase_from_dev": 0.03},
            "calibration_error": {"tolerance": 0.05},
        },
        "development_baseline": {
            "retail_pd_logit":    {"auc": 0.78, "ks": 0.42},
            "smb_pd_logit":       {"auc": 0.75, "ks": 0.38},
            "secured_lgd_frac":   {"pseudo_r2": 0.34, "rmse": 0.12},
            "unsecured_lgd_frac": {"pseudo_r2": 0.30, "rmse": 0.13},
        },
    }


def is_strictly_monotonic(values):
    """True if the series only ever goes up, or only ever goes down."""
    values = list(values)
    strictly_up   = all(b > a for a, b in zip(values, values[1:]))
    strictly_down = all(b < a for a, b in zip(values, values[1:]))
    return strictly_up or strictly_down


def print_summary(pd_metrics, lgd_metrics):
    """
    Print a short summary so we can confirm the designed story survived
    generation and rounding.
    """
    print("\n=== PD models ===")
    for model_id in ["MDL-A", "MDL-B"]:
        rows = pd_metrics[pd_metrics["model_id"] == model_id]
        over = rows[rows["psi"] > 0.25]
        first_breach = over["period"].iloc[0] if len(over) else "none"
        print(
            f"{model_id}: PSI min={rows['psi'].min():.3f} "
            f"max={rows['psi'].max():.3f} | first period PSI > 0.25: {first_breach}"
        )

    print("\n=== LGD models ===")
    for model_id in ["MDL-C", "MDL-D"]:
        rows = lgd_metrics[lgd_metrics["model_id"] == model_id]
        mono_psi  = is_strictly_monotonic(rows["psi"])
        mono_cal  = is_strictly_monotonic(rows["calibration_error"])
        mono_rmse = is_strictly_monotonic(rows["rmse"])
        print(
            f"{model_id}: PSI min={rows['psi'].min():.3f} "
            f"max={rows['psi'].max():.3f} | "
            f"max calibration_error={rows['calibration_error'].max():.3f} | "
            f"monotonic psi={mono_psi} calib={mono_cal} rmse={mono_rmse}"
        )

    print(
        "\nStory check: MDL-B should breach PSI at 2025-08; "
        "MDL-D should breach nothing but show monotonic=True on all three."
    )


def main():
    # One shared random generator, seeded, so noise is reproducible.
    rng = np.random.default_rng(SEED)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    registry    = build_registry()
    pd_metrics  = build_pd_metrics(rng)
    lgd_metrics = build_lgd_metrics(rng)
    policy      = build_policy_rules()

    registry.to_csv(DATA_DIR / "model_registry.csv", index=False)
    pd_metrics.to_csv(DATA_DIR / "pd_metrics.csv", index=False)
    lgd_metrics.to_csv(DATA_DIR / "lgd_metrics.csv", index=False)
    with open(DATA_DIR / "policy_rules.json", "w", encoding="utf-8") as f:
        json.dump(policy, f, indent=2)

    print(f"Wrote 4 files to {DATA_DIR}")
    print(f"  model_registry.csv  ({len(registry)} rows)")
    print(f"  pd_metrics.csv      ({len(pd_metrics)} rows)")
    print(f"  lgd_metrics.csv     ({len(lgd_metrics)} rows)")
    print(f"  policy_rules.json")

    print_summary(pd_metrics, lgd_metrics)


if __name__ == "__main__":
    main()
