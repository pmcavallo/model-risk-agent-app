# Build Prompt — Layer 3 fixes (parser greedy-match + step budget)

Two fixes after the Part B run. In Claude Code (in `model-risk-agent-app`):
`Read build-prompts/03b-tools-fixes-prompt.md and apply it.`

Keep the existing "before" traces (`test_results_toolstrap.txt`, `test_results_partb_*.txt`) —
they are teaching material. Save the new runs under new names (below).

---

## The prompt

```
Two fixes to src/agent.py. Keep the code clean and commented.

FIX 1 — the ACTION parser truncates arguments that contain a ")". In the Part B run, the agent
called write_finding with a summary that contained "(unsecured_lgd_frac)", and the non-greedy
regex  ACTION:\s*(\w+)\s*\((.*?)\)  stopped at the FIRST ")", mangling the argument.
  - Fix it so the argument capture runs to the LAST ")" on that line. The safest approach:
      * take only the FIRST line of the reply that begins with ACTION (case-insensitive),
      * match the tool name, then capture everything between the first "(" and the LAST ")"
        on that line (greedy), e.g. rfind(")") or a greedy regex anchored to end-of-line.
  - Keep the existing forgiving argument handling (strip a "name=" prefix, strip quotes/spaces,
    split top-level args on commas). Do NOT split on commas that are inside the summary text if
    that is easy to avoid; if it is not easy, it is fine for write_finding to take the whole
    remaining string as the summary.
  - Add a brief comment: a tool argument containing a ")" broke our hand-rolled parser — this is
    exactly the kind of thing the API's native tool-calling handles for you (Layer 3 -> next).

FIX 2 — a broad question (e.g. the committee question) hit MAX_STEPS with no FINAL answer,
because the well-designed agent makes granular per-metric check_policy / trend_facts calls.
  - Raise MAX_STEPS from 15 to 25 so a 4-model sweep can finish.
  - Add a brief comment: granular tool calls are good for SHOWING reasoning, but they cost steps;
    in production you would let the agent batch calls or raise the budget. (Do NOT add batching
    now — the granular calls make the reasoning visible in the demo.)

Do not change tools.py, the data, the model string, or anything else about the loop.
```

---

## Re-run and save (project venv)
```
.venv\Scripts\python.exe src\agent.py "Is model D actually fine, or is something building?"        > test_results_v3_modelD.txt
.venv\Scripts\python.exe src\agent.py "I have a model committee meeting in an hour. What are the three things I most need to know?"  > test_results_v3_committee.txt
.venv\Scripts\python.exe src\agent.py "If I could revalidate only one model this quarter, which and why?"  > test_results_v3_revalidate.txt
```
(If your shell mangles UTF-8 on redirect, run each and copy the console output into the file, as
before.)

## What to send me (to write the Layer 3 slide script from real runs)
- `test_results_v3_modelD.txt` — the key one. Confirm the agent uses check_policy + trend_facts
  and reasons to the MDL-D "building risk" conclusion ITSELF (not from a tool verdict).
- `test_results_v3_committee.txt` — confirm it now completes with a FINAL answer (no step-limit
  cutoff), and note how many steps it took (the step-budget teaching point).
- Confirm write_finding now logs the full summary (no truncation at the paren).

## For the slides (both fixes are teaching points)
- FIX 1: a tool argument with a ")" broke the hand-rolled parser. Same lesson as Layer 2's
  parser stumble, one more vote for native tool-calling. Keep the before trace.
- FIX 2: doing it right costs steps. Granular per-metric calls make reasoning visible but eat the
  budget. "Right design" vs "cheap passthrough" is a real tradeoff — name it.
