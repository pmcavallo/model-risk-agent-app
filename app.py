"""
Layer 4 — the Streamlit UI.

WHAT THIS IS: a small web app that runs the SAME hand-rolled agent from
src/agent.py and shows its reasoning live in the browser, step by step, then the
final finding. The terminal version (python src/agent.py "...") still works
unchanged; this is just a second way to SEE the same loop.

HOW IT WORKS: src/agent.py exposes a generator, agent_steps(question), that
yields one small event dict per step (think / action / observe / final). This
file walks that generator and draws each event as it arrives. Because the loop
is a generator, the page fills in live as the agent works — you watch it think,
call a tool, read the result, and decide — instead of staring at a spinner and
getting everything at the end.

TEACHING POINT: notice we did NOT rewrite the agent to make a UI. One loop, two
front ends. The agent's logic has a single home (src/agent.py); the terminal and
this app just choose how to display the events it yields. That separation is the
whole reason the refactor was worth it.

Run it:
  .venv\\Scripts\\streamlit run app.py
"""

import os
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# src/ lives next to this file; add it to the import path so we can import the
# agent loop. (Keeping app.py at the project root and the agent in src/ is a
# common, tidy layout.)
SRC_DIR = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SRC_DIR))

from agent import agent_steps, DEFAULT_MODEL, MAX_STEPS  # noqa: E402  (import after path setup)

# Resolve which model will actually run, the SAME way the loop does: the
# AGENT_MODEL env override wins, otherwise the code default. This is only for the
# caption; the loop resolves it again itself. load_dotenv makes .env visible here.
load_dotenv()
ACTIVE_MODEL = os.environ.get("AGENT_MODEL", DEFAULT_MODEL)


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Model Risk Analyst Agent",
    page_icon="🔎",
    layout="centered",
)

st.title("Model Risk Analyst Agent")
st.caption(
    f"A hand-rolled AI agent that reasons over credit-risk model metrics.  "
    f"Model: {ACTIVE_MODEL} · step budget: {MAX_STEPS}"
)

# A few example questions students (and you, in the demo) can click to run.
# The last one is the money question — the sub-threshold drift case.
EXAMPLES = [
    "Which of my models needs attention?",
    "Is model D actually fine, or is something building?",
    "Give me a portfolio health summary across all four models.",
    "Does MDL-B breach any policy thresholds? Which, and when?",
]


# ---------------------------------------------------------------------------
# Small helpers to draw each kind of event.
#
# We keep the drawing in one place so the run loop below stays readable. Each
# helper takes the event dict and renders it. Nothing here is agent logic — it
# is purely presentation.
# ---------------------------------------------------------------------------
def draw_think(event):
    """The agent's reasoning for this step (its full text reply)."""
    st.markdown(f"**Step {event['step']} — thinking**")
    st.markdown(event["reply"])


def draw_action(event):
    """A labeled box showing which tool the agent chose to call."""
    args = ", ".join(str(a) for a in event["args"])
    st.info(f"🔧 Calling tool: `{event['tool']}({args})`")


def draw_observe(event):
    """The tool's result, shown as monospace so tables line up."""
    st.markdown("**Observation** (what the tool returned):")
    st.code(event["observation"], language="text")


def format_cost_line(event):
    """A compact running-cost caption for one step.

    Teaching point made visible: the input-token count climbs every step because
    the whole conversation is re-sent each time. Students watch the dollar figure
    tick up as the agent works.
    """
    return (
        f"💰 step {event['step']}: "
        f"+{event['step_input_tokens']:,} in / +{event['step_output_tokens']:,} out tok  ·  "
        f"running total ${event['running_cost']:.4f} "
        f"({event['total_input_tokens']:,} in + {event['total_output_tokens']:,} out)"
    )


def run_and_stream(question):
    """
    Walk the agent generator and render each event live.

    st.status gives us a collapsible "working..." panel; we write each step into
    it as it arrives, then show the final answer outside the panel so it stands
    out. Because agent_steps() is a generator, this for-loop receives events one
    at a time, as the agent produces them — that is what makes it feel live.
    """
    final_answer = None
    hit_error = None
    final_cost = None  # (steps_used, input_tokens, output_tokens, dollars)

    with st.status("Agent is working…", expanded=True) as status:
        for event in agent_steps(question):
            kind = event["type"]

            if kind == "error":
                hit_error = event["message"]
                break
            elif kind == "start":
                st.write(f"Question: *{event['question']}*")
            elif kind == "think":
                draw_think(event)
            elif kind == "action":
                draw_action(event)
            elif kind == "observe":
                draw_observe(event)
            elif kind == "cost":
                st.caption(format_cost_line(event))
            elif kind == "final":
                final_answer = event["answer"]
                final_cost = (
                    event["steps_used"],
                    event["total_input_tokens"],
                    event["total_output_tokens"],
                    event["total_cost"],
                )
            elif kind == "stopped":
                final_cost = (
                    event["max_steps"],
                    event["total_input_tokens"],
                    event["total_output_tokens"],
                    event["total_cost"],
                )
                status.update(
                    label=f"Stopped at the {event['max_steps']}-step limit.",
                    state="error",
                )

        if final_answer is not None:
            status.update(label="Done — finding ready.", state="complete")

    if hit_error:
        st.error(hit_error)
        return

    if final_answer is not None:
        st.subheader("Finding")
        st.success(final_answer)

    # Show the run's total cost as a metric row (works for both a finished run
    # and one that hit the step limit). Makes "every step is a paid call" real.
    if final_cost is not None:
        steps_used, in_tok, out_tok, dollars = final_cost
        c1, c2, c3 = st.columns(3)
        c1.metric("Steps", steps_used)
        c2.metric("Tokens (in + out)", f"{in_tok + out_tok:,}")
        c3.metric("Est. cost", f"${dollars:.4f}")
        st.caption(
            "Cost is estimated from current per-token prices in src/agent.py "
            "(verify rates at docs.claude.com). Token counts are exact, from the API."
        )


# ---------------------------------------------------------------------------
# The page: an input box, example buttons, and the streamed run.
# ---------------------------------------------------------------------------
# Keep the chosen question in session state so example buttons and the text box
# cooperate (clicking an example fills the box; editing the box overrides it).
if "question" not in st.session_state:
    st.session_state.question = EXAMPLES[0]

st.markdown("**Ask the agent a question about the models:**")

# Example buttons, laid out in two rows of two.
cols = st.columns(2)
for i, example in enumerate(EXAMPLES):
    if cols[i % 2].button(example, use_container_width=True):
        st.session_state.question = example

question = st.text_input(
    "Question",
    value=st.session_state.question,
    label_visibility="collapsed",
)

run_clicked = st.button("Run the agent", type="primary")

if run_clicked:
    if question.strip():
        run_and_stream(question.strip())
    else:
        st.warning("Type a question (or click an example) first.")

# A short footer that reinforces the teaching point for anyone poking around.
st.divider()
st.caption(
    "This UI runs the exact same agent as `python src/agent.py \"...\"`. "
    "The loop lives once in `src/agent.py` and yields its steps; the terminal "
    "prints them, this page renders them. One agent, two front ends."
)
