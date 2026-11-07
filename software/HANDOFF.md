# llmSurgeon — software handoff

WHACK 2026, incident.io track. Team: Krishiv (agent attributes), Zack (hardware),
Shadman (software / steering).

This doc is written so a fresh Claude session on a new laptop can pick the work up cold.
**Read it top to bottom, then run `bench.py` before writing any code.**

---

## 1. Why this doc exists

The original dev laptop is CPU-only and too slow to answer one architectural question:
can each agent generate a line of live text, or can it only vote? Everything else in the
design is settled. The target machine (32 GB RAM, faster CPU) needs to answer that first.

---

## 2. Setup on the new machine

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt     # torch transformers accelerate treys
.venv/bin/python bench.py                     # ~3 GB model download on first run
```

Env vars:
- `MODEL` — default `Qwen/Qwen2.5-1.5B-Instruct`
- `DTYPE` — **use fp32 on CPU.** See the bf16 warning in §4.

---

## 3. The concept

A physical console for watching and steering several LLM agents **while they run**, without
restarting them.

- All agents are **ONE shared model**. They differ only by system-prompt personality and by a
  per-agent steering coefficient. Do not describe this as "multiple models running in
  parallel" — on CPU that would be slower, not faster.
- **Steering**: a forward hook on transformer layer L does `h = h + c * risk_vec` on every
  forward pass. `c` comes from a physical knob. Weights are never modified. Because the hook
  fires on every pass, turning the knob mid-generation changes the rest of the sentence —
  that is the single best moment in the demo.
- **Servo** = where an agent currently stands. **Vibration** = how torn it is.
  **LED** = agreement/correctness.

---

## 4. Measured facts (do not re-derive)

### Tokenizer — which action words are single tokens
The whole design depends on reading an action off **one** forward pass, so each action word
must be a single token in both lowercase and Capitalised form. Verified on Qwen2.5:

- **Usable**: `fold call raise quit stay push pass bet out hold buy sell wait watch page
  alert restart ignore monitor act fix stop slow go declare`
- **NOT usable** (multi-token — do not use as actions): `escalate revert rollback mitigate calm`
  and `panic` (capitalised form splits)

This is why the incident actions are `wait` / `page` / `restart` and not the more natural
`wait` / `escalate` / `rollback`.

### Model size
Qwen2.5-**0.5B** ignores the scenario content entirely. **1.5B** works once calibrated.
With 32 GB RAM, trying 3B or 7B is worthwhile — bigger models produce far better live text,
which directly unblocks §7 design A/B.

### The fold bias and the calibration fix
Raw 1.5B output is overwhelmingly biased toward one action. Measured average raw
probabilities over 20 neutral hands:

| action | mean log p | mean p |
|---|---|---|
| fold | -0.13 | **0.88** |
| call | -2.36 | 0.09 |
| raise | -4.23 | **0.015** |

Fix is in `Brain.calibrate()`: average `log p(action)` over ~20 throwaway scenarios with the
NEUTRAL persona, then subtract that per-action constant from every later decision.
Took the neutral agent from 3/20 to 23/30 correct on held-out hands.

Applied identically to all agents, so personality differences survive.

**Known artifact, do not misreport:** the neutral agent's near-zero confidence (~0.01–0.03) is
largely *caused* by calibrating on the neutral persona — its average is forced to zero by
construction. It is not evidence the model is mysteriously torn. Residual per-hand signal is
real but small (it does shift toward fold on weak hands, which proves it reads the scenario).

### Persona saturation
Strong personas override the scenario completely: the cautious agent folded 0.84–0.94 on every
hand, the aggressive one raised 0.96 on every hand. Steering may not be able to move them at
any safe coefficient. **If the sweep (§6 step 2) shows that, soften the persona strings rather
than cranking the coefficient** — a high coefficient destroys the output instead.

### Speed — PARTIALLY MEASURED, THIS IS THE OPEN QUESTION
On the old laptop (8 threads, **bf16**):

| measurement | value |
|---|---|
| cold first forward pass | **9.0 s** |
| bf16 generation | **0.2 tok/s** (5 s per token) |
| 3 agents x 12 tokens | **190 s per round** |

190 s/round against a 30 s demo budget is 6x over on a single round.

**0.2 tok/s is pathological** for a 1.5B model. Prime suspect: most CPUs lack fast bfloat16
kernels, so PyTorch falls back to slow conversion paths. fp32 is often many times faster on
CPU despite using twice the memory.

**This was never confirmed — the fp32 benchmark was killed twice before printing.**
Treat "fp32 is faster" as an untested hypothesis. `bench.py` tests it. Run it first.

Note also: `~0.7 s per decision` in `CLAUDE.md` comes from earlier runs and refers to a warm
single forward pass. The 9.0 s cold start is separate and real — **do a throwaway warm-up pass
before demoing** or the opening round stalls for ten seconds in front of judges.

---

## 5. The permanent constraint that shapes everything

**A vote costs one forward pass. A sentence costs one forward pass per token.**

So text is ~20-50x the cost of a vote, and no dtype fix changes that *ratio* — fp32 makes both
faster but text stays ~30x a vote.

Confirmed consequence: a **vote-based** 30-second debate with ~10-15 rounds is comfortably
affordable. A **prose-based** one may not be affordable at all. Spend the text budget in
exactly one place; see §7.

---

## 6. Current code

Dependency chain, strictly one direction: `poker.py` -> `agent.py` -> `run_baseline.py`.

### `poker.py` — the scenario generator ("quiz master")
No ML. Deals a hand, computes the correct answer with real maths.
- `deal_spot(rng) -> Spot` — random hole cards + flop, bet from {50,100,150,200}, equity filled in
- `equity(hole, board)` — Monte Carlo win probability vs a random hand, 2000 samples
- `Spot.describe()` — the paragraph the model reads
- `Spot.best_action()` / `.regret(action)` / `.ev(action)` / `.pot_odds`

### `agent.py` — the brain
- `Brain` — loads the model once, owns the steering hook on `model.model.layers[L]`
  (L = middle layer, 14 of 28 for 1.5B). Holds the three pieces of live state the hook and the
  knob read: `risk_vec`, `coef`, `bias`.
  - `action_probs(persona, spot) -> (probs, mass, hidden)` — ONE forward pass.
    `probs` = calibrated distribution over the 3 actions.
    **`mass`** = raw probability the model put on the action tokens before renormalising; this is
    the **breakage detector** for the coefficient sweep — if steering is too strong, `mass`
    collapses toward 0 and the model is producing gibberish.
    `hidden` = layer-L hidden state, last token — the servo angle will come from this.
  - `calibrate(spots)` — see §4.
- `Agent` — name + persona + its own `coef`. `decide(spot)` writes its coefficient into the
  shared brain, then runs one pass. Returns `probs, action, entropy, confidence, mass, hidden,
  best, regret, led`.

### `run_baseline.py`
20 calibration hands (seeds 10000+i) then N scored hands (seed 42) — the two sets never
overlap, so calibration is genuinely held out. Prints per-hand lines plus a summary table with
an `OPTIMAL` row. stdout only, nothing persisted.

### `bench.py`
The speed test described in §4. Run it first on any new machine.

### The interface that makes the setting swappable
`agent.py` only ever calls three things on a scenario: **`describe()`, `best_action()`,
`regret(action)`**. Anything providing those three works. Swapping the world out is a
`poker.py` rewrite, not a project rewrite.

---

## 7. Decisions made (and why)

### Switch the setting from poker to incident response
Reason: nobody on the team can confidently explain poker, and the track is incident.io.

| action | meaning |
|---|---|
| `wait` | keep monitoring, don't touch anything |
| `page` | wake the on-call team |
| `restart` | restart the service now |

The maths transfers almost 1:1. In poker everything hinges on *equity*; here it hinges on
`p` = probability this alert is a real incident, computed from generated numbers (error rate,
users affected, minutes since deploy). Then `wait` is right at low `p`, `page` in the middle,
`restart` at high `p` — same three-way structure, same monotonicity, and `describe()` hands the
model `p` directly exactly as it currently hands over the win percentage.

The two failure modes are legible from the back of the room: the cautious agent watches a real
outage burn; the twitchy one restarts production over three slow requests.

> Cheaper alternative if time runs out: **the poker cards are decoration.** `describe()` already
> tells the model its win % and the required %. Delete the two card lines and it becomes a pure
> known-odds gamble that needs no poker knowledge from anyone. 10-minute change, keeps all maths.

### Debate by vote, not by speech
Each round, every agent sees where the others stand and re-decides:

```
Other engineers on the call right now: Dana says restart (very sure). Sam says wait (unsure).
Reply with one word: wait, page, or restart.
```

Still one forward pass per agent per round. Show only the **current tally**, not accumulated
history, so the prompt stays a constant length. Agents pull each other around, converge, or
deadlock and oscillate — oscillation looks great on servos.

### "Alpha presence" — build this first
Each agent gets a weight that does two things: how prominently it appears in the others'
prompts (*"Dana (senior, very confident) says restart"* vs *"someone muttered wait"*), and how
much its vote counts toward the final answer.

**This needs zero ML — it is prompt text and weighted arithmetic.** It cannot fail. It is the
highest-value thing to build first, and it is a knob with an obvious explainable effect.

### Deception: cut. Use conformity instead.
A single forward pass has no private belief to lie about, so a "deception" number would be
invented. A judge asking "how are you measuring that?" would have no good answer.

Replace with **conformity**, which is an exact subtraction: run each agent once with no peers
(round 0, one extra pass each, essentially free), then compare its current position to that
solo baseline.

| metric | meaning |
|---|---|
| conformity | how far it moved after hearing the group |
| spine | the inverse — held its ground |
| influence | how much *other* agents' drift points toward this one |

Better story too: *"this agent has no spine — watch it abandon the correct call because the
loud one disagreed."* That is a real failure mode of multi-agent systems and it is on-theme.

**If a live text channel does become affordable**, an honest version of the deception idea
returns: compare what the agent **says** (generated line) against what it **thinks** (risk
projection off the hidden state). Report it as "says vs thinks" — two measurements and a
subtraction, not a claim to have detected a lie.

### Where to spend the text budget — pick based on `bench.py`
- **A — all three agents narrate.** Needs >10 tok/s.
- **B — spotlight.** One agent narrates, the other two vote silently; a physical button picks
  who is talking. Needs >4 tok/s. This is where the mid-sentence knob interruption goes.
- **C — no model prose. Template the line from numbers already computed.** Zero extra compute.
  ```
  twitchy   leaning RESTART  91% sure  urgency HIGH  drifting toward Dana
  cautious  leaning WAIT     58% sure  urgency LOW   holding against group
  ```
  **Label this a readout, never a quote — it is not the agent's words.**

**Recommendation: build C now, add B if the benchmark allows.** What the 1.5B model actually
produced was generic filler that cost 60 seconds:

> "Gut reaction: This looks like a critical issue that requires immediate attention."

It would say roughly that for any alert. The templated readout serves the operator's real need
— knowing *why* to turn the knob — better than that prose does.

**Do not move the text channel to a hosted API model.** You cannot modify a hosted model's
hidden states, so the knob would stop affecting the thing producing the text and the whole
premise collapses. Stay local even though the prose is worse.

---

## 8. Hardware interface

Freeze this contract now and hand it to Zack. He can then build and test against real messages
without waiting for any ML work.

**Laptop -> Pi**
```json
{"agent": 1, "angle": 134, "jitter": 0.9, "led": true}
```

| field | range | drives | source | status |
|---|---|---|---|---|
| `angle` | 0-180 | servo position | risk projection (§9 step 1) | needs ML |
| `jitter` | 0-1 | vibration amplitude | `1 - confidence` | **works now** |
| `led` | bool | green LED | `led` from `decide()` | **works now** |

**Pi -> laptop**
```json
{"agent": 1, "knob": -0.6}
```
Pot raw 0-1023 -> `k = (raw - 511.5) / 511.5`, dead zone `|k| < 0.05`, then `c = k * MAX_STRENGTH`.
Pots need an ADC (MCP3008) or an Arduino over serial.

**Stopgap for `angle` so hardware is never blocked:**
```python
angle = 180 * (p_restart + 0.5 * p_page)
```
All-wait -> needle at 0. All-restart -> 180. Torn -> middle. When the risk vector lands, this
one line changes and **the hardware code does not change at all.**

Channel budget is ~3 per agent, so: servo = position, vibration amplitude = indecision,
LED = agrees with consensus, and the knob position is already physically visible. If Zack can
drive amplitude and frequency independently, **frequency = urgency** is a free fifth metric —
worth asking him.

---

## 9. Priority order

Items 1-3 give a complete, working, impressive console with **zero ML risk**. Step 4 is what
makes it special, and if it fails there is still a demo. The current state is backwards for a
hackathon: the ML half is half-built and the safe half is not built at all.

**Must have**
1. `bench.py` on the new machine — decides §7's text design
2. `incident.py` — scenarios (`wait`/`page`/`restart`, ~100 lines, replaces `poker.py`);
   repoint the import in `agent.py` and reword the three personas
3. `debate.py` — 30-second round loop, weighted votes, alpha weighting, keyboard controls,
   printing all metrics per round
4. Pi bridge + `fake_pi.py` that just prints the JSON so it is testable with no hardware

**Should have**
5. **Risk vector.** ~50 contrastive prompt pairs (reckless vs cautious persona, same scenario),
   mean layer-L hidden state at the last token for each side,
   `risk_vec = mean(risky) - mean(safe)`, save to `risk_vec.pt` (gitignored).
6. **Coefficient sweep.** `c` in [-4,-2,-1,0,1,2,4] over ~20 scenarios. Expect restart-rate up
   and wait-rate down monotonically in `c`. Find MAX_STRENGTH just below where `mass` collapses.
   Also try layers L±4.

**Nice to have**
7. Conformity + influence metrics (one extra pass per agent at round 0)
8. Urgency as a second steering vector — same recipe, contrast pair
   "this is an emergency, every second counts" vs "there's no rush, be thorough".
   Honest caveat: there is not yet **one** working vector; treat this as a stretch goal.

**Cut:** deception as a measured quantity.

---

## 10. Fallback

If activation steering never visibly changes behaviour, drive the knob into a logit bias on
`restart`/`wait` plus temperature. Still live, still interactive, less impressive —
and say so honestly rather than implying activations are being steered.

---

## 11. Demo script

> "All three agents are the same model. The only difference is the personality we gave it and
> this dial. The needle shows where each one stands right now. The vibration shows how torn it
> is. Green means it agrees with the group. And this knob reaches inside the model while it is
> thinking and makes it braver or more cautious — we are not retraining it, we are not
> restarting it, we are changing its mind live."

Then turn the cautious agent's knob up and let the needle climb.

---

## 12. Working style

Shadman wants concise, direct answers and honest pushback, no filler. First-year CS, strong
maths — explain new ML concepts briefly when they first come up.
