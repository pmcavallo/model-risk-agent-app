# Model Risk Analyst Agent

An AI **agent** that acts as a model risk analyst. Ask it open-ended questions about a set of
credit-risk models and it reasons over their monitoring metrics — deciding which tools to call, in
what order, branching on model type, and composing a judgment — then writes a finding.

It is built as a **from-scratch, hand-rolled agent loop**: no LangChain, no LangGraph, no framework.
The model returns plain text; the code parses it, runs a tool, feeds the result back, and repeats.
Every step is visible, so you can watch the agent think → act → observe → decide.

> Teaching project for **BUAN 6V99 — Agentic AI & Process Automation** (UT Dallas). The point is to
> show what a task that *genuinely needs an agent* looks like — not a task a script could do.
> All data is **synthetic**; no real models, portfolios, or institutional data are used.

---

## The question that proves the point

> *"Is model D actually fine, or is something building?"*

Model D breaches no threshold all year — but every one of its metrics drifts the same direction for
twelve straight months. A script that only checks thresholds says "OK." The agent, reasoning over
the *trajectory*, flags the building risk. That gap — between "no rule broken" and "something is
clearly wrong" — is why this task needs an agent and not a script.

---

## What it does

- Answers open-ended questions about four credit-risk models (2 PD, 2 LGD) by calling tools.
- Branches on model type: PD models are judged on AUC / KS / PSI; LGD models on
  pseudo-R² / RMSE / calibration error / PSI.
- Uses tools for the deterministic, precision-critical work (exact threshold checks, trend facts)
  and keeps the **judgment** — what matters, how severe, what to prioritize — for the agent itself.
- Runs in two ways from **one** shared loop: a terminal script and a live Streamlit web app.

## The four models (a designed story)

| Model | Type | Behaviour over 12 months |
|---|---|---|
| MDL-A `retail_pd_logit` | PD | Stable and healthy. |
| MDL-B `smb_pd_logit` | PD | Obvious breach: PSI crosses 0.25, AUC falls below the floor. |
| MDL-C `secured_lgd_frac` | LGD | Stable and healthy. |
| MDL-D `unsecured_lgd_frac` | LGD | **The discriminating case** — never breaches, but drifts monotonically the wrong way. |

---

## Run it locally

**Prerequisites:** Python 3.10+ and an Anthropic API key.

```bash
# 1. Clone and enter the project
git clone https://github.com/pmcavallo/model-risk-agent-app.git
cd model-risk-agent-app

# 2. Create a virtual environment and install dependencies
python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # macOS / Linux
pip install -r requirements.txt

# 3. Add your API key
copy .env.example .env            # Windows  (cp on macOS/Linux)
# then edit .env and set ANTHROPIC_API_KEY=sk-ant-...

# 4. (Optional) regenerate the synthetic data — it is deterministic (seed=42)
python src/generate_data.py

# 5a. Run in the terminal
python src/agent.py "Is model D actually fine, or is something building?"

# 5b. Or run the web app
streamlit run app.py
```

The terminal and the web app run the **same agent**. The loop lives once in `src/agent.py` and
yields its steps; the terminal prints them, the app renders them live. One agent, two front ends.

---

## How it is built (five layers)

1. **Data** (`src/generate_data.py`) — deterministic synthetic monitoring data (registry, PD/LGD
   metrics, policy thresholds), designed so MDL-B obviously breaches and MDL-D quietly drifts.
2. **Agent core** (`src/agent.py`) — the hand-rolled loop. The model replies `ACTION: tool(args)`
   or `FINAL: ...`; the code parses it, runs the tool, feeds back the result, repeats, up to a step
   cap. Exposed as a generator (`agent_steps`) so both front ends can consume it.
3. **Tools** (`src/tools.py`) — plain functions: `list_models`, `get_pd_metrics`, `get_lgd_metrics`,
   `check_policy`, `trend_facts`, `write_finding`. Tools fetch and compute; they do not decide
   whether a model is "fine."
4. **UI** (`app.py`) — a Streamlit app that streams the agent's reasoning live, then the finding.
5. **Ship** — this README, a public GitHub repo, and free hosting on Streamlit Community Cloud.

```
model-risk-agent-app/
├── app.py                  # Streamlit UI (Layer 4)
├── src/
│   ├── generate_data.py    # Layer 1 — synthetic data
│   ├── agent.py            # Layer 2 — the hand-rolled loop (generator + terminal view)
│   └── tools.py            # Layer 3 — the agent's tools
├── data/                   # generated CSVs + policy_rules.json (source data, committed)
├── build-prompts/          # the prompt files that built each layer
├── test_results*.txt        # captured runs — the traces the build was debugged from
├── .streamlit/config.toml  # UI theme
├── requirements.txt
└── .env.example            # copy to .env and add your key
```

---

## Deploy to Streamlit Community Cloud (free)

1. Push this repo to GitHub (public or private — Community Cloud supports both).
2. Go to **share.streamlit.io**, sign in with GitHub, and click **New app**.
3. Point it at this repo, branch `main`, main file `app.py`.
4. Under **Advanced settings → Secrets**, add your key:
   ```
   ANTHROPIC_API_KEY = "sk-ant-..."
   ```
   (Community Cloud injects this as an environment variable — the app reads it the same way it reads
   your local `.env`.)
5. Deploy. The committed `data/` CSVs ship with the repo, so the hosted app has everything it needs.

**Never commit your real `.env`.** It is git-ignored on purpose; the key lives in Community Cloud's
Secrets, not in the repo.

---

## Notes

- **Model string:** `claude-sonnet-5`. Model names change — verify current strings at
  docs.claude.com if the API rejects it.
- **Cost:** each step is one API call. A full multi-model run is a few cents on Sonnet. The loop
  cap (`MAX_STEPS`) is also a cost cap.
- **Synthetic data only.** Nothing here is real model, portfolio, or institutional data.

---

*BUAN 6V99 — Agentic AI & Process Automation · Paulo Cavallo, PhD · UT Dallas*
