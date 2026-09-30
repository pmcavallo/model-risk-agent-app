# Build Prompt — Layer 3: The Tools (facts in tools, judgment in the agent)

Two parts. PART A builds a deliberately "too-smart" tool to show the passthrough trap live
(great teaching moment). PART B deletes it and builds the real design.
In Claude Code (in `model-risk-agent-app`): read this file and apply PART A first, test, then PART B.

The headline lesson: **tools do deterministic, precision-critical work; the agent keeps the
judgment.** A tool that does everything turns the agent into a pointless passthrough.

---

## PART A — the "too-smart" tool (build to show the trap)

```
In the model-risk-agent project, add ONE tool to src/tools.py called assess_model(model_id). It
should do EVERYTHING: read the model's metrics, check every metric against policy_rules.json,
look at the 12-month trends, decide a severity, and return a finished plain-English finding with a
recommendation. Then update the agent's system prompt (src/agent.py) to mention only two tools:
list_models and assess_model.

I am doing this on purpose to show students that when a tool does all the work, the agent becomes
a pointless passthrough. Keep it simple. We delete it in the next step.
```

Test (project venv):
```
.venv\Scripts\python.exe src\agent.py "Which of my models needs attention, and what should I do?"
```
**Watch:** the agent just calls assess_model on each model and repeats its output. It is not
reasoning. Save this run as `test_results_toolstrap.txt`. **That is the trap.** Then do PART B.

---

## PART B — the real tool design (facts + checks in tools, judgment in the agent)

```
In the model-risk-agent project, refactor the tools. DELETE assess_model. The design rule: tools
do deterministic, precision-critical work; the AGENT keeps the judgment (what matters, severity,
prioritization). Keep code clean, commented, beginner-readable.

src/tools.py should expose these tools, each with a precise docstring (the docstring is how the
agent decides when to call it — write it as if it were a prompt):

  list_models()
    -> the registry: for each model, id, name, type (PD or LGD).  [already exists]

  get_pd_metrics(model_id)
    -> 12 monthly rows (period, auc, ks, psi, default rates) for a PD model.  [already exists]

  get_lgd_metrics(model_id)
    -> 12 monthly rows (period, pseudo_r2, rmse, calibration_error, psi, mean_lgd_actual,
       mean_lgd_predicted) for an LGD model.  [already exists]

  check_policy(model_type, metric, value)   [NEW]
    -> read policy_rules.json and return the status for ONE value:
         psi: "ok" (<0.10), "monitor" (0.10-0.25), "breach" (>0.25)
         floor metrics (auc, ks, pseudo_r2): "ok" if value >= floor else "breach"
         ceiling metric (rmse): "ok" if value <= ceiling else "breach"
         calibration_error: "ok" if value <= tolerance else "breach"
       Return the status AND the threshold it compared against. Do NOT return advice or a
       recommendation. Docstring must tell the agent: "Use this instead of comparing values to
       thresholds yourself." (Reason: the model is unreliable at exact numeric comparison; the
       tool makes it exact.)

  trend_facts(model_id, metric)   [NEW]
    -> deterministic FACTS about that metric's 12-month series: first value, last value, net
       change, direction ("up"/"down"/"flat"), strictly_monotonic (True/False), and the longest
       run of consecutive steps in the drift direction. Return FACTS ONLY — no verdict, no
       "this is concerning". Docstring: "Use to understand a metric's trajectory over time."

  write_finding(model_id, severity, summary)   [NEW]
    -> the agent passes severity ("breach"/"watch"/"ok") and a one-paragraph summary. This tool
       formats a finding record, appends it to data/findings.json (create if missing), and
       returns a confirmation string. The tool does NOT decide severity — the agent does.

UPDATE src/agent.py:
  - Remove the hardcoded policy thresholds from the system prompt.
  - Instead instruct the agent: use check_policy for any threshold question, and trend_facts for
    any trajectory question. The rules now live in policy_rules.json, reached only through
    check_policy.
  - Update the TOOLS section of the system prompt to list all six tools with a one-line
    description each (matching the docstrings).
  - Keep the hand-rolled loop, the hardened forgiving parser, visible step printing, MAX_STEPS,
    and the key-from-environment behavior exactly as they are.

Do not change the data files.
```

Test the interrogation (project venv), and SAVE each run:
```
.venv\Scripts\python.exe src\agent.py "I have a model committee meeting in an hour. What are the three things I most need to know?"
.venv\Scripts\python.exe src\agent.py "Is model D actually fine, or is something building?"
.venv\Scripts\python.exe src\agent.py "If I could revalidate only one model this quarter, which and why?"
```

## What to send me to verify
- `test_results_toolstrap.txt` (PART A — the passthrough).
- The trace for "Is model D actually fine, or is something building?" (PART B).
- Confirm: does the agent now call check_policy for exact statuses and trend_facts for the
  trajectory, instead of eyeballing the tables? Does it still flag MDL-D correctly?

## Why this is the lesson (for the slides)
- PART A: the tool did everything → the agent did nothing. The passthrough trap, seen live.
- PART B: facts and single-value checks live in tools; the agent sequences them, branches, and
  makes the call. Reasoning returns.
- check_policy also moves the RULES out of the prompt into one auditable place — the governance
  lesson (and it fixes the Layer 2 "agent eyeballed the thresholds" limitation).
- trend_facts reports the drift as facts; the agent decides it MATTERS. Facts in the tool,
  judgment in the agent — the exact line this whole layer teaches.
