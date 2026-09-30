# Build Prompt — Layer 2 fixes (max_tokens + parser hardening)

Two changes to the Layer 2 agent, after the first live run. Both are real teaching moments.
In Claude Code (in `model-risk-agent-app`): `Read build-prompts/02b-agent-fixes-prompt.md and apply it.`

Keep the original trace (`test_results.txt`) — it is the "before" we teach from.

---

## The prompt

```
Make two changes to src/agent.py. Keep the code clean and commented.

CHANGE 1 — the FINAL answer is being truncated (it cut off mid-sentence and never reached the
last model). Raise the response token limit so the agent can finish its assessment.
  - In the client.messages.create(...) call, set max_tokens to 2000 (up from whatever it is now).
  - Add a brief comment explaining that a low max_tokens truncates the final answer.

CHANGE 2 — make the argument parser more forgiving. In the first live run the model wrote
  ACTION: get_pd_metrics(model_id="MDL-A")
and the parser treated the whole string  model_id="MDL-A  as the id, which failed twice before
the model fell back to get_pd_metrics(MDL-A). Fix the parser so it handles keyword-style and
quoted arguments:
  - After extracting the text inside the parentheses, for each argument:
      * if it contains an "=", keep only the part AFTER the "=" (drops a "model_id=" prefix)
      * strip surrounding single or double quotes
      * strip whitespace
  - So all of these resolve to MDL-A:
      MDL-A
      "MDL-A"
      'MDL-A'
      model_id="MDL-A"
      model_id='MDL-A'
  - Keep the existing behavior: case-insensitive ACTION:, FINAL: detection, and returning a
    gentle correction when nothing parses.
  - Add a short comment noting this is defensive parsing, and that this fragility is exactly why
    the API's native tool-calling exists (which we cover in Layer 3).

Do not change tools.py or the data. Do not change the model string.
```

---

## Re-test (project venv)
```
.venv\Scripts\python.exe src\agent.py "Which of my models needs attention, and what should I do?"
```

## What to confirm in the new trace
- The FINAL answer runs to completion (no mid-sentence cutoff) and **reaches MDL-D**.
- The agent gets each model's metrics in ONE clean step now (no more failed retries on the
  keyword-argument form).
- It still branches PD vs LGD correctly.
- Crucially: does it flag **MDL-D** — no breach, but monotonic drift on every metric — as a
  "watch / building risk," distinct from MDL-B's outright breach? That is the money moment.

Save the new run as `test_results_v2.txt` and send it over.

---

## Why both changes are teaching moments (for the slides)
- **max_tokens truncation:** a real, common gotcha. The agent "stopped thinking" — but it did
  not; the response was cut off. Teaches students to recognize truncation vs. a reasoning failure.
- **Parser hardening:** the model phrased a tool call in a reasonable way our hand-rolled parser
  did not expect. It recovered, but clumsily. This is the live case for why structured
  tool-calling exists. Keep the before/after: fragile parser (original trace) -> hardened parser
  (clean trace) -> "and here's how the API does it for you" (Layer 3).
