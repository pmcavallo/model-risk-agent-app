"""
Layer 3 preview — the agent's tools.

A "tool" here is just a plain Python function the agent can ask us to run. Each
one reads a CSV from data/ and returns a short, human-readable STRING (never a
DataFrame) so the text can be fed straight back to the model.

Design rule for the whole project: tools do the deterministic, precision-critical
work (look up the data, pull the exact numbers). The AGENT keeps the judgment
(what matters, how severe it is, what to do about it). So these functions fetch
and format — they do not decide whether a model is "fine."
"""

import json
from pathlib import Path

import pandas as pd

# data/ lives one level up from this src/ folder.
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def list_models():
    """Return the model registry: each model's id, name, and type (PD or LGD)."""
    registry = pd.read_csv(DATA_DIR / "model_registry.csv")

    lines = ["Models under monitoring:"]
    for _, row in registry.iterrows():
        lines.append(
            f"  {row['model_id']} | {row['model_name']} | type={row['model_type']}"
        )
    return "\n".join(lines)


def _lookup_model(model_id):
    """
    Small helper: find one model in the registry by id.

    Returns the matching row, or None if the id isn't found. Used by both metric
    tools so they can fail gracefully on a bad id.
    """
    registry = pd.read_csv(DATA_DIR / "model_registry.csv")
    match = registry[registry["model_id"] == model_id]
    if len(match) == 0:
        return None
    return match.iloc[0]


def get_pd_metrics(model_id):
    """
    Return 12 months of PD metrics (AUC, KS, PSI, default rates) for a PD model.

    PD models are judged on discrimination (AUC, KS) and drift (PSI).
    """
    model = _lookup_model(model_id)
    if model is None:
        return f"No model with id '{model_id}'. Call list_models() to see valid ids."
    if model["model_type"] != "PD":
        return (
            f"{model_id} is a {model['model_type']} model, not PD. "
            f"Use get_lgd_metrics for LGD models (AUC/KS do not apply to LGD)."
        )

    metrics = pd.read_csv(DATA_DIR / "pd_metrics.csv")
    rows = metrics[metrics["model_id"] == model_id]

    lines = [f"PD metrics for {model_id} ({model['model_name']}), 12 months:"]
    lines.append("  period    auc    ks     psi    default_rate(actual/expected)")
    for _, r in rows.iterrows():
        lines.append(
            f"  {r['period']}  {r['auc']:.3f}  {r['ks']:.3f}  {r['psi']:.3f}  "
            f"{r['default_rate_actual']:.2f} / {r['default_rate_expected']:.2f}"
        )
    return "\n".join(lines)


def get_lgd_metrics(model_id):
    """
    Return 12 months of LGD metrics (pseudo-R2, RMSE, calibration error, PSI)
    for an LGD model.

    LGD models are judged on fit (pseudo-R2, RMSE), calibration
    (calibration_error), and drift (PSI). AUC and KS do NOT apply.
    """
    model = _lookup_model(model_id)
    if model is None:
        return f"No model with id '{model_id}'. Call list_models() to see valid ids."
    if model["model_type"] != "LGD":
        return (
            f"{model_id} is a {model['model_type']} model, not LGD. "
            f"Use get_pd_metrics for PD models."
        )

    metrics = pd.read_csv(DATA_DIR / "lgd_metrics.csv")
    rows = metrics[metrics["model_id"] == model_id]

    lines = [f"LGD metrics for {model_id} ({model['model_name']}), 12 months:"]
    lines.append("  period    pseudo_r2  rmse   calib_error  psi")
    for _, r in rows.iterrows():
        lines.append(
            f"  {r['period']}  {r['pseudo_r2']:.3f}      {r['rmse']:.3f}  "
            f"{r['calibration_error']:.3f}        {r['psi']:.3f}"
        )
    return "\n".join(lines)


# =====================================================================
# The three "right design" tools.
#
# Design rule for the whole layer: tools do the deterministic,
# precision-critical work; the AGENT keeps the judgment (what matters, how
# severe, what to prioritize). Contrast with the deleted assess_model, which
# did everything and turned the agent into a passthrough.
# =====================================================================


def check_policy(model_type, metric, value):
    """
    Return the policy status of ONE metric value, plus the threshold it was
    compared against. Reads the thresholds from policy_rules.json.

    Use this instead of comparing values to thresholds yourself: the language
    model is unreliable at exact numeric comparison, and this tool makes it
    exact and auditable.

    Returns a status ("ok", "monitor", or "breach") and the threshold used.
    It does NOT recommend an action - deciding what to do is your job.
    """
    policy = json.loads((DATA_DIR / "policy_rules.json").read_text())

    section = model_type.strip().lower()
    if section not in ("pd", "lgd"):
        return f"Unknown model_type '{model_type}'. Use 'pd' or 'lgd'."
    rules = policy[section]

    metric = metric.strip().lower()
    if metric not in rules:
        valid = ", ".join(k for k in rules.keys())
        return f"No policy for metric '{metric}' under {section}. Known: {valid}."

    # value arrives as text from the parser, e.g. "0.200" - convert to a number.
    try:
        value = float(value)
    except (TypeError, ValueError):
        return f"Could not read '{value}' as a number."

    rule = rules[metric]

    # PSI has three bands; everything else is a single floor / ceiling / tolerance.
    if metric == "psi":
        if value < rule["ok_max"]:
            status = "ok"
        elif value <= rule["monitor_max"]:
            status = "monitor"
        else:
            status = "breach"
        return (
            f"{section} {metric}={value:.3f} -> {status} "
            f"(bands: ok<{rule['ok_max']}, monitor {rule['ok_max']}-{rule['monitor_max']}, "
            f"breach>{rule['monitor_max']})"
        )

    if "floor" in rule:  # auc, ks, pseudo_r2 - higher is better
        status = "ok" if value >= rule["floor"] else "breach"
        return f"{section} {metric}={value:.3f} -> {status} (floor {rule['floor']})"

    if "ceiling" in rule:  # rmse - lower is better
        status = "ok" if value <= rule["ceiling"] else "breach"
        return f"{section} {metric}={value:.3f} -> {status} (ceiling {rule['ceiling']})"

    if "tolerance" in rule:  # calibration_error - lower is better
        status = "ok" if value <= rule["tolerance"] else "breach"
        return f"{section} {metric}={value:.3f} -> {status} (tolerance {rule['tolerance']})"

    return f"Metric '{metric}' has no comparable threshold in policy."


def _metric_series(model_id, metric):
    """
    Return the 12-month list of values for one metric of one model, or an error
    string. Picks the PD or LGD table automatically from the model's type.
    """
    model = _lookup_model(model_id)
    if model is None:
        return f"No model with id '{model_id}'. Call list_models() to see valid ids."

    filename = "pd_metrics.csv" if model["model_type"] == "PD" else "lgd_metrics.csv"
    rows = pd.read_csv(DATA_DIR / filename)
    rows = rows[rows["model_id"] == model_id]

    if metric not in rows.columns:
        valid = ", ".join(c for c in rows.columns if c not in ("model_id", "period"))
        return f"No metric '{metric}' for {model_id}. Available: {valid}."

    return list(rows[metric])


def trend_facts(model_id, metric):
    """
    Report deterministic FACTS about a metric's 12-month trajectory: first
    value, last value, net change, direction (up/down/flat), whether it is
    strictly monotonic, and the longest run of consecutive steps in the drift
    direction.

    Use to understand a metric's trajectory over time. Returns FACTS ONLY - no
    verdict, no "this is concerning." Whether a trajectory MATTERS is your call.
    """
    metric = metric.strip().lower()
    series = _metric_series(model_id, metric)
    if isinstance(series, str):  # an error message came back
        return series

    first = series[0]
    last = series[-1]
    net_change = last - first

    if net_change > 0:
        direction = "up"
    elif net_change < 0:
        direction = "down"
    else:
        direction = "flat"

    # Month-over-month steps.
    steps = [b - a for a, b in zip(series, series[1:])]
    strictly_monotonic = all(s > 0 for s in steps) or all(s < 0 for s in steps)

    # Longest run of consecutive steps that go in the overall drift direction.
    longest_run = 0
    current_run = 0
    for s in steps:
        moves_in_drift_direction = (
            (direction == "up" and s > 0) or (direction == "down" and s < 0)
        )
        if moves_in_drift_direction:
            current_run += 1
            longest_run = max(longest_run, current_run)
        else:
            current_run = 0

    return (
        f"{model_id} {metric}: first={first:.3f}, last={last:.3f}, "
        f"net_change={net_change:+.3f}, direction={direction}, "
        f"strictly_monotonic={strictly_monotonic}, "
        f"longest_run_in_direction={longest_run} of {len(steps)} steps"
    )


def write_finding(model_id, severity, *summary_parts):
    """
    Record a finding you have decided on. YOU pass the severity ("breach",
    "watch", or "ok") and a one-paragraph summary; this tool just formats the
    record, appends it to data/findings.json (creating the file if needed), and
    returns a confirmation. It does NOT decide severity - that judgment is yours.
    """
    # The forgiving parser splits arguments on commas, so a summary paragraph
    # arrives as several pieces. Re-join them back into one sentence.
    summary = ", ".join(part for part in summary_parts if part)

    findings_path = DATA_DIR / "findings.json"

    # Load whatever is already there (or start a fresh list).
    if findings_path.exists():
        findings = json.loads(findings_path.read_text())
    else:
        findings = []

    findings.append(
        {
            "model_id": model_id,
            "severity": severity.strip().lower(),
            "summary": summary,
        }
    )

    findings_path.write_text(json.dumps(findings, indent=2))
    return (
        f"Finding recorded for {model_id} (severity={severity.strip().lower()}). "
        f"data/findings.json now holds {len(findings)} finding(s)."
    )


# A simple name -> function map so the agent loop can dispatch by tool name.
TOOLS = {
    "list_models": list_models,
    "get_pd_metrics": get_pd_metrics,
    "get_lgd_metrics": get_lgd_metrics,
    "check_policy": check_policy,
    "trend_facts": trend_facts,
    "write_finding": write_finding,
}
