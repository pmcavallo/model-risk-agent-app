"""
Layer 2 — the agent core: a from-scratch, hand-rolled agent loop.

THE TEACHING POINT: we do NOT use LangChain, LangGraph, or the Anthropic SDK's
built-in tool-calling. The model returns plain TEXT; OUR code parses that text,
runs a tool, and feeds the result back. The loop is visible and every step is
emitted, so students can watch the agent think -> act -> observe -> think again.

The contract we ask the model to follow (see SYSTEM_PROMPT):
  - To use a tool, reply with exactly one line:   ACTION: tool_name(args)
  - When ready to answer, reply with:             FINAL: your answer

ONE LOOP, TWO INTERFACES (Layer 4 note):
  The loop lives in `agent_steps()`, a generator that YIELDS one event per step
  instead of printing. Two thin consumers drive off the same generator:
    - `run_agent()` prints the events -> the terminal demo (unchanged experience).
    - app.py (the Streamlit UI) renders the events live in the browser.
  This is the clean way to make one agent power both a terminal and a UI: the
  loop has a single source of truth, and each interface just decides how to SHOW
  the events. No duplicated logic, no capturing of printed text.

Run it (terminal):
  .venv\\Scripts\\python.exe src\\agent.py "Which of my models needs attention?"
"""

import os
import re
import sys

from dotenv import load_dotenv
import anthropic

# Our tools live next door in tools.py.
from tools import TOOLS

# The model string. Sonnet is a good balance of speed and reasoning for a demo.
# verify current model string at docs.claude.com
#
# CONFIG FROM THE ENVIRONMENT (same idea as the API key): the DEFAULT lives here
# in code, but you can override it without editing code by setting AGENT_MODEL in
# your .env (e.g. AGENT_MODEL=claude-haiku-4-5-20251001 to try the cheaper model).
# A sensible default in code + an env override is how real projects handle config.
# We read the override at call time (inside agent_steps, after load_dotenv) so a
# .env change takes effect without touching this file.
DEFAULT_MODEL = "claude-sonnet-5"

# Pricing for the cost tracker, in dollars PER MILLION TOKENS (input, output) --
# output is the pricier side. Rates as of Aug 2026. PRICES CHANGE and differ per
# model -- verify current rates at docs.claude.com before trusting the dollars.
# (Token COUNTS are always exact, from the API; only the dollar conversion uses
#  these constants.) The tracker picks the row matching whichever model runs, so
# switching to Haiku reports HAIKU's price, not Sonnet's -- an honest comparison.
PRICING_PER_MTOK = {
    # model string : (input $/Mtok, output $/Mtok)
    "claude-sonnet-5": (2.00, 10.00),
    "claude-haiku-4-5-20251001": (1.00, 5.00),
    "claude-opus-5": (5.00, 25.00),
}
# Fallback used if the running model isn't in the table above (so an unknown or
# newer model still runs; the cost line just flags that the price is a guess).
FALLBACK_PRICE_PER_MTOK = (2.00, 10.00)


def _prices_for(model):
    """Return (input_price, output_price) per Mtok for a model, or the fallback."""
    return PRICING_PER_MTOK.get(model, FALLBACK_PRICE_PER_MTOK)

# Safety valve: never loop forever. Real agents always cap their steps.
# Raised 15 -> 25 -> 35: the well-designed agent makes granular per-metric
# check_policy / trend_facts calls, so the BROADEST question ("which models need
# attention?") requires a full 4-model sweep -- ~26-30 steps to investigate every
# model AND still have room to write findings and give a FINAL answer. At 25 the
# agent did all the right analysis but ran out of budget one step before it could
# conclude. 35 gives the widest question honest headroom while still being a real
# cap (a runaway loop is still stopped). Granular calls are good for SHOWING
# reasoning, but they cost steps; in production you would let the agent batch
# calls or raise the budget further. (We keep calls granular so reasoning stays visible.)
MAX_STEPS = 35

# The system prompt is the agent's "job description." It is used VERBATIM.
SYSTEM_PROMPT = """You are a model risk analyst assistant. You answer questions about a set of credit-risk models
by using tools. You cannot see any data directly - you must use tools to get it.

TOOLS
- list_models(): returns the model registry: id, name, and type (PD or LGD).
- get_pd_metrics(model_id): returns 12 months of AUC, KS, PSI, and default rates for a PD model.
- get_lgd_metrics(model_id): returns 12 months of pseudo-R2, RMSE, calibration error, and PSI for an LGD model.
- check_policy(model_type, metric, value): returns the policy status ("ok"/"monitor"/"breach") of
  one metric value and the threshold it was compared against.
- trend_facts(model_id, metric): returns facts about a metric's 12-month trajectory (first, last,
  net change, direction, strictly_monotonic, longest run in the drift direction).
  Use the EXACT metric names (not the shortened column headers you see in the tables):
    PD models:  auc, ks, psi, default_rate_actual, default_rate_expected, volume
    LGD models: pseudo_r2, rmse, calibration_error, psi, mean_lgd_actual, mean_lgd_predicted, volume
  (e.g. it is "calibration_error", not "calib_error"; "default_rate_actual", not "default_rate".)
- write_finding(model_id, severity, summary): records a finding you have decided on.

HOW TO REPLY
- To use a tool, reply with EXACTLY one line:
  ACTION: tool_name(arguments)
- When you can answer, reply with:
  FINAL: your answer

RULES
- One action per turn. Wait for the OBSERVATION before choosing the next action.
- If you do not know what models exist, call list_models() first.
- PD and LGD models are judged by DIFFERENT metrics. Use get_pd_metrics for PD models and
  get_lgd_metrics for LGD models. Never judge an LGD model with AUC or KS.
- For ANY threshold question, call check_policy - do NOT compare numbers to thresholds yourself.
  You are unreliable at exact numeric comparison; the tool and the policy file are the source of truth.
- For ANY trajectory or "is something building" question, call trend_facts. A metric can stay
  inside its threshold yet drift steadily - the facts reveal that; you decide whether it matters.
- YOU decide severity and prioritization. The tools give you facts and single-value statuses;
  the judgment (what matters, how severe, what to do) is yours.
- Be efficient. You do not need to check every metric of every model to answer. Gather enough
  evidence to justify your conclusion, then STOP investigating and give your answer.
- Always finish with a FINAL answer. Do not spend your last step on another tool call: once you
  have what you need, write any findings and then reply with FINAL: summarizing what matters and
  what to do. An investigation with no FINAL answer is a failure, however good the analysis was.
- Think step by step. Be specific about which model, which metric, which period."""


def get_model_reply(client, messages, model):
    """
    Send the full conversation to the model and return (reply_text, usage).

    We ask for plain text (no native tool-calling). The response may contain a
    few content blocks; we keep the text ones and join them. We ALSO return the
    token usage the API reports for this call, so the loop can track cost: every
    call bills input tokens (everything we sent) + output tokens (what came back).
    """
    response = client.messages.create(
        model=model,
        # A low max_tokens silently TRUNCATES the reply: the final assessment
        # gets cut off mid-sentence and looks like the agent "stopped thinking."
        # 2000 gives it room to finish a full multi-model finding.
        max_tokens=2000,
        system=SYSTEM_PROMPT,
        messages=messages,
    )
    text_parts = [block.text for block in response.content if block.type == "text"]
    reply = "".join(text_parts).strip()
    # response.usage.input_tokens / .output_tokens are the EXACT counts the API
    # billed for this one call. We hand them back so the loop can sum them.
    usage = {
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
    }
    return reply, usage


def _clean_arg(arg):
    """
    Normalize one argument. The model may phrase it several reasonable ways our
    hand-rolled parser didn't originally expect, e.g.
        MDL-A   "MDL-A"   'MDL-A'   model_id="MDL-A"   model_id='MDL-A'
    all of which should resolve to  MDL-A.
    """
    # If it's a keyword form (model_id="MDL-A"), keep only what's after the "=".
    if "=" in arg:
        arg = arg.split("=", 1)[1]
    # Drop surrounding quotes and whitespace.
    return arg.strip().strip("'\"")


def parse_action(reply):
    """
    Try to read a line of the form:  ACTION: tool_name(args)

    Returns (tool_name, [args]) if it parses, or None if it doesn't. The parser
    is deliberately forgiving: case-insensitive "ACTION:", strips quotes and
    spaces around each argument, and drops a "name=" prefix.
    """
    # Look only at the FIRST line that starts with ACTION (case-insensitive).
    action_line = None
    for line in reply.splitlines():
        if line.strip().upper().startswith("ACTION:"):
            action_line = line
            break
    if action_line is None:
        return None

    # Read the tool name and the opening parenthesis.
    match = re.search(r"action:\s*([a-zA-Z_]\w*)\s*\(", action_line, re.IGNORECASE)
    if not match:
        return None
    tool_name = match.group(1)

    # Capture everything between the FIRST "(" and the LAST ")" on the line.
    # Greedy on purpose: a tool argument may itself contain ")", e.g. a summary
    # like "(unsecured_lgd_frac)". Our earlier non-greedy match stopped at the
    # first ")" and truncated the argument — exactly the kind of fragility the
    # API's native tool-calling handles for you, which is where Layer 3 heads next.
    open_paren = action_line.index("(")
    close_paren = action_line.rfind(")")
    if close_paren <= open_paren:
        return None
    raw_args = action_line[open_paren + 1:close_paren].strip()

    # No args at all, e.g. list_models()
    if raw_args == "":
        return tool_name, []

    # write_finding's last argument is a free-text summary that can itself
    # contain commas. Split off only model_id and severity, then keep the WHOLE
    # remaining string as one summary argument so the text stays intact.
    if tool_name == "write_finding":
        parts = raw_args.split(",", 2)  # at most 3 pieces: id, severity, summary
        return tool_name, [_clean_arg(part) for part in parts]

    # Every other tool takes simple arguments: split on commas and clean each.
    return tool_name, [_clean_arg(arg) for arg in raw_args.split(",")]


def run_tool(tool_name, args):
    """Run the named tool with its args and return the resulting string."""
    if tool_name not in TOOLS:
        valid = ", ".join(TOOLS.keys())
        return f"Unknown tool '{tool_name}'. Valid tools are: {valid}."
    try:
        return TOOLS[tool_name](*args)
    except Exception as error:
        # Tools should be robust, but if one throws, tell the agent instead of crashing.
        return f"Tool '{tool_name}' raised an error: {error}"


# =====================================================================
# THE LOOP — as a generator.
#
# This is the same hand-rolled loop as before; the only change is that it now
# YIELDS a small dict describing each step instead of printing. Yielding lets
# ANY caller decide how to show the step: the terminal prints it, the Streamlit
# UI renders it in the browser. One loop, many interfaces.
#
# Each yielded event is a plain dict with a "type" so the caller knows what it
# is. The types, in the order they occur within a step:
#   {"type": "start",  "model": ..., "max_steps": ..., "question": ...}
#   {"type": "think",  "step": n, "reply": "<the agent's full reply text>"}
#   {"type": "action", "step": n, "tool": "check_policy", "args": [...]}
#   {"type": "observe","step": n, "observation": "<tool result text>"}
#   {"type": "cost",   "step": n, per-step + running token/dollar totals}
#   {"type": "final",  "answer": ..., + the run's total tokens and cost}
#   {"type": "stopped","max_steps": ..., + the run's total tokens and cost}
# A "think" with no following "action" (parse failed) is followed by an
# "observe" whose observation explains the format problem — same as before.
#
# COST NOTE (a real teaching point): every step is one API call, and each call
# re-sends the WHOLE conversation so far as input. So input tokens GROW every
# step — step 20 pays to re-read everything from steps 1-19. That is why agent
# loops cost more than people expect, and why tight prompts and a step cap matter.
# =====================================================================
def agent_steps(question):
    """Run the hand-rolled loop, yielding one event dict per step.

    This is a GENERATOR: it pauses after each yield and resumes when the caller
    asks for the next event. That is what lets the UI show steps live, as they
    happen, instead of waiting for the whole run to finish.
    """
    load_dotenv()  # reads ANTHROPIC_API_KEY from a local .env file
    if not os.environ.get("ANTHROPIC_API_KEY"):
        yield {
            "type": "error",
            "message": "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key.",
        }
        return

    client = anthropic.Anthropic()  # picks up the key from the environment

    # Resolve the model: env override (AGENT_MODEL) wins, else the code default.
    # load_dotenv() above has already pulled .env into the environment, so a
    # student can switch models by editing .env alone -- no code change.
    model = os.environ.get("AGENT_MODEL", DEFAULT_MODEL)
    price_in, price_out = _prices_for(model)

    yield {"type": "start", "model": model, "max_steps": MAX_STEPS, "question": question}

    # The conversation starts with the user's question.
    messages = [{"role": "user", "content": question}]

    # Cost accumulators — summed across every API call in this run.
    total_input_tokens = 0
    total_output_tokens = 0

    for step in range(1, MAX_STEPS + 1):
        reply, usage = get_model_reply(client, messages, model)

        # Add this call's tokens to the running totals, then compute running cost
        # (tokens / 1e6 * price-per-million, input and output priced separately).
        # Prices are for the model actually running (see _prices_for), so a Haiku
        # run is costed at Haiku's rate, not Sonnet's.
        total_input_tokens += usage["input_tokens"]
        total_output_tokens += usage["output_tokens"]
        running_cost = (
            total_input_tokens / 1_000_000 * price_in
            + total_output_tokens / 1_000_000 * price_out
        )

        # The agent "thinks" out loud: yield its full reply for this step.
        yield {"type": "think", "step": step, "reply": reply}

        # Emit this step's tokens plus running totals so a caller can show cost
        # ticking up live. Watch step_input_tokens climb: that is the whole
        # conversation being re-sent every step.
        yield {
            "type": "cost",
            "step": step,
            "step_input_tokens": usage["input_tokens"],
            "step_output_tokens": usage["output_tokens"],
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "running_cost": running_cost,
        }

        # Did the agent give its final answer?
        if "FINAL:" in reply.upper():
            final = reply[reply.upper().index("FINAL:") + len("FINAL:"):].strip()
            yield {
                "type": "final",
                "answer": final,
                "total_input_tokens": total_input_tokens,
                "total_output_tokens": total_output_tokens,
                "total_cost": running_cost,
                "steps_used": step,
            }
            return

        # Otherwise, try to read and run an action.
        parsed = parse_action(reply)
        if parsed is None:
            observation = (
                "I could not find a valid action. Reply with EXACTLY one line in the "
                "form  ACTION: tool_name(args)  or give your answer as  FINAL: ..."
            )
        else:
            tool_name, args = parsed
            # Announce the action BEFORE running it, so the UI can show
            # "calling check_policy(...)" while the tool runs.
            yield {"type": "action", "step": step, "tool": tool_name, "args": args}
            observation = run_tool(tool_name, args)

        # Show what came back from the tool (or the format-error message).
        yield {"type": "observe", "step": step, "observation": observation}

        # Feed the round back into the conversation:
        #   - the agent's own reply (assistant turn)
        #   - the observation, marked so the agent knows it's a tool result (user turn)
        messages.append({"role": "assistant", "content": reply})
        messages.append({"role": "user", "content": "OBSERVATION: " + observation})

    # Fell out of the loop without a FINAL answer.
    yield {
        "type": "stopped",
        "max_steps": MAX_STEPS,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_cost": running_cost,
    }


# =====================================================================
# TERMINAL CONSUMER — prints the yielded events.
#
# This reproduces the original terminal experience exactly: it just walks the
# generator and prints each event. The agent's logic is untouched; this is only
# a way to DISPLAY the events on a console.
# =====================================================================
def run_agent(question):
    """The terminal view: run the loop and print every step as it happens."""
    for event in agent_steps(question):
        kind = event["type"]

        if kind == "error":
            print("ERROR: " + event["message"])
            return

        if kind == "start":
            print("=" * 70)
            print(
                f"MODEL RISK ANALYST AGENT  |  model={event['model']}  "
                f"|  max steps={event['max_steps']}"
            )
            print(f"Question: {event['question']}")
            print("=" * 70)

        elif kind == "think":
            print(f"\n----- STEP {event['step']} -----")
            print("Agent:")
            print(event["reply"])

        elif kind == "cost":
            # One compact line per step: this step's tokens + the running cost.
            print(
                f"[cost] step {event['step']}: "
                f"in {event['step_input_tokens']:,} / out {event['step_output_tokens']:,} tok  "
                f"| running total {event['total_input_tokens']:,} in + "
                f"{event['total_output_tokens']:,} out  = ${event['running_cost']:.4f}"
            )

        elif kind == "observe":
            print("\nObservation:")
            print(event["observation"])

        elif kind == "final":
            print("\n" + "=" * 70)
            print("FINAL ANSWER:")
            print(event["answer"])
            print("-" * 70)
            print(
                f"Run cost: {event['steps_used']} steps, "
                f"{event['total_input_tokens']:,} input + "
                f"{event['total_output_tokens']:,} output tokens  "
                f"= ${event['total_cost']:.4f}"
            )
            print("=" * 70)

        elif kind == "stopped":
            print("\n" + "=" * 70)
            print(f"Stopped: hit the {event['max_steps']}-step limit without a FINAL answer.")
            print("-" * 70)
            print(
                f"Run cost: {event['max_steps']} steps, "
                f"{event['total_input_tokens']:,} input + "
                f"{event['total_output_tokens']:,} output tokens  "
                f"= ${event['total_cost']:.4f}"
            )
            print("=" * 70)

        # Note: we do not print the "action" event in the terminal, because the
        # agent's reply (the "think" event) already shows the ACTION: line. The
        # UI uses the "action" event to render a labeled tool-call box; the
        # terminal doesn't need it. Same events, each interface shows what fits.


if __name__ == "__main__":
    # Take the question from the command line, or ask for it if none was given.
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        question = input("Ask the model risk analyst a question: ")
    run_agent(question)
