# Build Prompt — Layer 4: the Streamlit UI

**How to use this in class:** point Claude Code at this file — "Read
`build-prompts/04-ui-prompt.md` and build it." This is the same workflow as
Layers 1-3: the prompt is the spec, the agent does the build, you verify.

---

## Goal

Add a small Streamlit web UI that runs the existing hand-rolled agent and shows
its reasoning **live**, step by step, then the final finding. The terminal
version (`python src/agent.py "..."`) must keep working exactly as before.

## The one important design decision

Do **not** write a second copy of the agent loop for the UI, and do **not** have
the UI capture the terminal's printed text. Instead, make the loop **one source
of truth that both the terminal and the UI consume**:

1. Refactor the loop in `src/agent.py` into a **generator**, `agent_steps(question)`,
   that **yields one small event dict per step** instead of printing.
2. Keep `run_agent(question)` as a thin **terminal consumer** that walks
   `agent_steps(...)` and prints each event — so the terminal experience is
   unchanged.
3. Write `app.py` (Streamlit) as a second consumer that walks the same generator
   and renders each event in the browser.

This is the teaching point of the layer: **one agent, two front ends.** The loop
logic lives in exactly one place; each interface only decides how to *show* the
events.

## The event contract

`agent_steps(question)` yields dicts, each with a `"type"`:

- `{"type": "start", "model", "max_steps", "question"}` — once, at the top.
- `{"type": "think", "step", "reply"}` — the agent's full text reply for a step.
- `{"type": "action", "step", "tool", "args"}` — the tool it chose (yielded
  *before* the tool runs, so the UI can show "calling …").
- `{"type": "observe", "step", "observation"}` — the tool's returned text (or a
  format-error message if the reply didn't parse).
- `{"type": "final", "answer"}` — the agent's final answer; loop ends.
- `{"type": "stopped", "max_steps"}` — only if it never said FINAL.
- `{"type": "error", "message"}` — e.g. missing API key.

Do not change the agent's actual logic (the parser, the tools, the system
prompt, MAX_STEPS, max_tokens). Only change *how the loop emits* each step
(print → yield) and add the two consumers.

## What `app.py` should do

- Title + a one-line caption (model name, step budget).
- A text box for the question, plus 3-4 **example buttons**. Include the money
  question: *"Is model D actually fine, or is something building?"*
- A "Run the agent" button.
- When run, walk `agent_steps(...)` and render live inside a `st.status(...)`
  panel:
  - `think` → the step number and the agent's reasoning text.
  - `action` → a labeled box: "Calling tool: `tool_name(args)`".
  - `observe` → the tool result in a monospace/code block so tables line up.
  - `final` → show the answer prominently **outside** the status panel (e.g.
    `st.success`), under a "Finding" heading.
  - `stopped` / `error` → show clearly.
- Keep it **beginner-readable** (this is read aloud in class): short helper
  functions, clear names, comments that explain *why*, not just *what*.

## Layout / files

- `app.py` at the project root.
- The agent stays in `src/agent.py`; `app.py` adds `src/` to the import path and
  imports `agent_steps`, `MODEL`, `MAX_STEPS`.
- Add `.streamlit/config.toml` for a clean theme (optional but nice).
- Add `streamlit` to `requirements.txt`.

## Definition of done

- `python src/agent.py "Which of my models needs attention?"` prints the same as
  before (unchanged terminal experience).
- `streamlit run app.py` opens a page; asking the money question streams the
  agent's steps live and ends with the MDL-D "building risk" finding.
- No duplicated loop logic; the UI and terminal both drive off `agent_steps`.

## After it works

Update `CLAUDE.md` "Current status": check off Layer 4, note `app.py` exists and
that the loop is now a generator with two consumers. Then Layer 5 (README +
GitHub + host on Streamlit Community Cloud) is the last step.
