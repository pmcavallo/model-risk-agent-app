# Build Prompt — Layer 1: Synthetic Data

Paste this into Claude Code (working in this `model-risk-agent-app` folder) to build the data
generator. Then run `python src/generate_data.py` and verify the story.

---

## The prompt

```
Create a small Python project that generates synthetic model-risk monitoring data for a
teaching demo in a graduate agentic-AI class. Keep the code clean, commented, and
beginner-readable, because I will explain it to students line by line.

PROJECT STRUCTURE
Create (if missing):
  data/                      (output goes here)
  src/generate_data.py       (the generator)
  requirements.txt           (pandas, numpy)

CONTEXT
We monitor 4 credit-risk models over 12 months (periods 2025-01 to 2025-12):
  - 2 PD models (probability of default), built with simple logistic regression
  - 2 LGD models (loss given default), built with fractional logistic regression
PD and LGD are validated with DIFFERENT metrics, so they go in separate tables.

USE A FIXED SEED (seed=42). Add small noise (auc/ks/psi about +/-0.004, rmse/calibration
about +/-0.002, rates +/-0.1). IMPORTANT: for model MDL-D the noise must be small enough that
every listed metric stays strictly MONOTONIC across the 12 months (see below) — that monotonic
drift is the whole point of the demo.

GENERATE FOUR FILES IN data/:

1) data/model_registry.csv  (4 rows)
   columns: model_id, model_name, model_type, method, deployed
   rows:
     MDL-A, retail_pd_logit,     PD,  simple logistic,     2024-12
     MDL-B, smb_pd_logit,        PD,  simple logistic,     2024-12
     MDL-C, secured_lgd_frac,    LGD, fractional logistic, 2024-12
     MDL-D, unsecured_lgd_frac,  LGD, fractional logistic, 2024-12

2) data/pd_metrics.csv  (2 PD models x 12 months = 24 rows)
   columns: model_id, period, auc, ks, psi, default_rate_actual, default_rate_expected, volume
   MDL-A (STABLE): auc ~0.780 flat; ks ~0.420 flat; psi rises 0.03 -> 0.08 (never reaches 0.10);
                   default_rate_actual ~3.0; default_rate_expected 3.0; volume ~20000
   MDL-B (OBVIOUS BREACH): psi per month = 0.04,0.06,0.07,0.09,0.12,0.16,0.20,0.27,0.31,0.35,0.38,0.41
                   auc per month = 0.752,0.749,0.746,0.742,0.735,0.727,0.716,0.703,0.692,0.681,0.671,0.661
                   ks 0.381 -> 0.271 ; default_rate_actual 3.5 -> 6.5 ; default_rate_expected 3.6 ; volume ~15000

3) data/lgd_metrics.csv  (2 LGD models x 12 months = 24 rows)
   columns: model_id, period, pseudo_r2, rmse, calibration_error, psi, mean_lgd_actual, mean_lgd_predicted, volume
   MDL-C (STABLE): pseudo_r2 ~0.340; rmse ~0.120; calibration_error ~0.010; psi 0.03 -> 0.08;
                   mean_lgd_actual ~0.40; mean_lgd_predicted ~0.40; volume ~8000
   MDL-D (MONOTONIC SUB-THRESHOLD DRIFT — breaches NOTHING, drifts EVERYWHERE):
                   psi per month               = 0.09,0.10,0.11,0.12,0.13,0.14,0.15,0.16,0.17,0.18,0.19,0.20  (monotonic up, stays < 0.25)
                   calibration_error per month = 0.010,0.013,0.017,0.021,0.025,0.029,0.033,0.037,0.040,0.043,0.046,0.048  (monotonic up, stays < 0.05)
                   rmse per month              = 0.130 -> 0.180 (monotonic up, stays < 0.20)
                   pseudo_r2 per month         = 0.300 -> 0.240 (monotonic DOWN, stays > 0.20)
                   mean_lgd_actual per month   = 0.55 -> 0.66 (monotonic up)
                   mean_lgd_predicted          = ~0.55 flat
                   volume ~9000

4) data/policy_rules.json  with exactly this content:
{
  "pd": {
    "psi":  { "ok_max": 0.10, "monitor_max": 0.25,
              "actions": { "ok": "No action", "monitor": "Investigate and document",
                           "breach": "Trigger revalidation" } },
    "auc":  { "floor": 0.70, "material_decline_from_dev": 0.05 },
    "ks":   { "floor": 0.30, "material_decline_from_dev": 0.10 },
    "default_rate": { "actual_vs_expected_tolerance_pct": 25 }
  },
  "lgd": {
    "psi":  { "ok_max": 0.10, "monitor_max": 0.25,
              "actions": { "ok": "No action", "monitor": "Investigate and document",
                           "breach": "Trigger revalidation" } },
    "pseudo_r2":         { "floor": 0.20, "material_decline_from_dev": 0.05 },
    "rmse":              { "ceiling": 0.20, "material_increase_from_dev": 0.03 },
    "calibration_error": { "tolerance": 0.05 }
  },
  "development_baseline": {
    "retail_pd_logit":    { "auc": 0.78, "ks": 0.42 },
    "smb_pd_logit":       { "auc": 0.75, "ks": 0.38 },
    "secured_lgd_frac":   { "pseudo_r2": 0.34, "rmse": 0.12 },
    "unsecured_lgd_frac": { "pseudo_r2": 0.30, "rmse": 0.13 }
  }
}

ROUNDING: auc/ks/psi/pseudo_r2/rmse/calibration_error to 3 decimals; rates and mean_lgd to
2 decimals; volume integer.

AFTER GENERATING, print a console summary:
  - PD: for each PD model, min/max PSI and the first period where PSI exceeds 0.25 (or "none").
  - LGD: for each LGD model, min/max PSI, max calibration_error, and whether psi,
    calibration_error, and rmse are each strictly monotonic across the 12 months (True/False).
This lets me confirm the story: MDL-B breaches PSI at 2025-08; MDL-D breaches nothing but shows
monotonic=True on psi, calibration_error, and rmse.

Keep it to pandas + numpy only.
```

---

## Verify after running
1. `model_registry.csv` = 4 rows (2 PD, 2 LGD); `pd_metrics.csv` = 24; `lgd_metrics.csv` = 24.
2. MDL-B PSI first exceeds 0.25 at **2025-08**; AUC drops below 0.70 ~month 10.
3. MDL-A PSI never reaches 0.10.
4. MDL-D: psi, calibration_error, rmse all **monotonic=True**, and none breaches its threshold.
5. MDL-C flat and healthy.

If any check fails, fix the generator (usually shrink the noise) before building the agent.
