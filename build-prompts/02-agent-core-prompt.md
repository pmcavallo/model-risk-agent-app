# Build Prompt — Layer 2: The Agent Core

Paste-free workflow: in Claude Code (working in `model-risk-agent-app`), just say:
`Read build-prompts/02-agent-core-prompt.md and build it.`
Run Layer 1 first so the `data/` files exist. Then test with the questions at the bottom.

---

## The prompt

```
Build a from-scratch AI agent in Python for a teaching demo. Do NOT use LangChain, LangGraph,
or the Anthropic SDK's built-in tool-calling. The whole point is that students SEE the loop, so
we hand-roll it: the model returns plain text, our code parses it, runs a tool, and feeds the
result back. Keep the code clean, commented, and beginner-readable.

The data files already exist in data/ (from Layer 1):
  data/model_registry.csv   columns: model_id, model_name, model_type, method, deployed
  data/pd_metrics.csv        columns: model_id, period, auc, ks, psi, default_rate_actual, default_rate_expected, volume
  data/lgd_metrics.csv       columns: model_id, period, pseudo_r2, rmse, calibration_error, psi, mean_lgd_actual, mean_lgd_predicted, volume

CREATE:
  src/tools.py     - three tools (below)
  src/agent.py     - the hand-rolled loop
  .env.example     - a template line: ANTHROPIC_API_KEY=your-key-here
  update requirements.txt - add: anthropic, python-dotenv  (keep pandas)
  (.gitignore already ignores .env, __pycache__/, .venv/, data/findings.json)

TOOLS (src/tools.py) — plain functions that read the CSVs with pandas and return a short,
readable STRING (not a dataframe). Each gets a clear docstring. Handle a bad model_id gracefully.
  - list_models() -> each model's id, name, and type (PD or LGD).
  - get_pd_metrics(model_id) -> the 12 monthly rows of auc, ks, psi, default rates for a PD model.
  - get_lgd_metrics(model_id) -> the 12 monthly rows of pseudo_r2, rmse, calibration_error, psi
    for an LGD model.

AGENT LOOP (src/agent.py):
  - Load ANTHROPIC_API_KEY from the environment via python-dotenv (reads a .env file).
    NEVER hard-code the key.
  - Set MODEL as a constant near the top. Use a current Claude Sonnet model string and add a
    comment: "# verify current model string at docs.claude.com".
  - Use the SYSTEM PROMPT below verbatim.
  - Keep a messages list; start with the user's question.
  - Loop up to MAX_STEPS = 15:
      * call the model with the full messages list
      * print the step number and the model's raw reply, clearly labeled (a separator line between steps)
      * if the reply contains "FINAL:", print the final answer and stop
      * else parse an action of the form  ACTION: tool_name(args)
          - forgiving parser: case-insensitive "ACTION:", regex to capture tool_name and the
            text in the parentheses; strip quotes/spaces; split args on commas
          - if parsing fails, set the observation to a gentle correction telling it to use the
            exact format ACTION: tool_name(args)
          - if it parses, dispatch to the matching tool and capture its returned string
      * print the observation, clearly labeled
      * append the model reply, and append the observation as a user message prefixed "OBSERVATION: "
  - If the loop ends without FINAL, print that it hit the step limit.
  - if __name__ == "__main__": read the question from a command-line argument, else input().

SYSTEM PROMPT (verbatim):
'''
You are a model risk analyst assistant. You answer questions about a set of credit-risk models
by using tools. You cannot see any data directly - you must use tools to get it.

TOOLS
- list_models(): returns the model registry: id, name, and type (PD or LGD).
- get_pd_metrics(model_id): returns 12 months of AUC, KS, PSI, and default rates for a PD model.
- get_lgd_metrics(model_id): returns 12 months of pseudo-R2, RMSE, calibration error, and PSI for an LGD model.

HOW TO REPLY
- To use a tool, reply with EXACTLY one line:
  ACTION: tool_name(arguments)
- When you can answer, reply with:
  FINAL: your answer

RULES
- One action per turn. Wait for the OBSERVATION before choosing the next action.
- PD and LGD models are judged by DIFFERENT metrics. Use get_pd_metrics for PD models and
  get_lgd_metrics for LGD models. Never judge an LGD model with AUC or KS.
- If you do not know what models exist, call list_models() first.
- Policy thresholds:
    PSI:  below 0.10 OK,  0.10 to 0.25 monitor,  above 0.25 breach (revalidation).
    PD:   AUC floor 0.70,  KS floor 0.30.
    LGD:  pseudo-R2 floor 0.20,  RMSE ceiling 0.20,  calibration error tolerance 0.05.
- Think step by step. Be specific about which model, which metric, which period.
'''

Print a one-line banner at start so a live audience knows what is running. Make the printed
steps easy to read.
```

---

## Test after building (use the project venv)
```
.venv\Scripts\python.exe src\agent.py "What models do I have?"
.venv\Scripts\python.exe src\agent.py "Which of my models needs attention, and what should I do?"
```
Watch the steps: it should call list_models, then branch, get_pd_metrics for the PD models and
get_lgd_metrics for the LGD ones, then give a FINAL answer.

## What to send me to verify
- The full printed step-by-step for the second question (the reasoning trace).
- Whether the parser held (no stalls), and whether it correctly used get_pd_metrics vs
  get_lgd_metrics per model type.

## Note (Layer 2 keeps thresholds in the prompt)
On purpose. Layer 3 moves the rules out of the prompt into a `check_policy` tool. For now the
thresholds live in the system prompt so the loop runs end to end with just the three data tools.
