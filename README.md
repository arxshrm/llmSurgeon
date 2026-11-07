# llmSurgeon

A physical console for watching and steering several LLM agents **while they run**, without
restarting them.

WHACK 2026, incident.io track. Krishiv (agent attributes), Zack (hardware),
Shadman (software / steering).

Three agents get the same on-call alert and must choose: **wait**, **page**, or **restart**.
They are all the same local model with the same prompt. The only thing that differs between
them is a knob.

- **Servo** = where that agent currently stands, read off its hidden state.
- **Vibration** = how torn it is (entropy over the three actions).
- **LED** = it picked the genuinely correct action.
- **Knob** = a coefficient `c`. A forward hook adds `c * risk_vec` to the hidden state at a
  middle layer on every forward pass. Nothing is retrained, nothing is restarted — turning
  the knob mid-run changes the agent's mind live.

The agents also argue: each round every agent sees where the others stand and re-decides, and
a loud agent (high "alpha presence") appears more prominently in everyone else's prompt and
counts for more in the vote. That makes the interesting failure modes visible — an agent with
no spine abandoning the correct call because the confident one disagreed.

## Start here
- **[software/HANDOFF.md](software/HANDOFF.md)** — the design, the measured facts, and what is
  still unknown. Source of truth.
- [software/CLAUDE.md](software/CLAUDE.md) — quick orientation and the command list.
- [idea.org](idea.org) — where the idea came from.

## Run it
```bash
cd software
python -m venv .venv
.venv/Scripts/pip install torch --index-url https://download.pytorch.org/whl/cu130
.venv/Scripts/pip install -r requirements.txt
.venv/Scripts/python tests.py      # 32 tests, no model download needed
.venv/Scripts/python debate.py     # the demo
```
`WORLD=poker python debate.py` runs the same console on poker hands instead of alerts.
