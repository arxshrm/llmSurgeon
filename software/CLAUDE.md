# llmSurgeon — software (Shadman)

WHACK 2026 hackathon, incident.io track. Team: Krishiv (agent attributes), Zack (hardware), Shadman (software: steering / "weight tweaking").

## Concept
Physical console for watching and steering several LLM agents live, without restarting them.
- Every agent gets the same poker spot: fold, call or raise?
- All agents = ONE shared LLM, different system-prompt personality + own steering coefficient.
- **Servo = gauge.** Angle = how risky the agent currently "thinks" (projection of hidden state onto the risk vector). Jitter amplitude = indecision (entropy over fold/call/raise).
- **Knob = steering.** Sets coefficient `c`; hidden state at layer L becomes `h + c * risk_vec` on every forward pass. c<0 cautious, 0 none, c>0 reckless. Weights never change.
- **Green LED** = agent chose the highest-EV action (NOT "won the hand" — that's luck).

## Files
- `poker.py` — deals a flop spot (2 hole cards + 3 board, pot 100, bet 50/100/150/200, stack 400). Equity via Monte Carlo vs random hand (`treys`). EV: fold=0, call=eq*(pot+2bet)-bet, raise(all-in, assume called)=eq*(pot+2stack)-stack. ⇒ fold if eq<pot odds, raise if eq>0.5, else call. `describe()` gives the LLM its win % and pot odds (small models can't compute equity).
- `agent.py` — `Brain` (shared model + forward hook on `model.model.layers[L]`, middle layer by default) and `Agent` (persona + coef). Decision = ONE forward pass, read next-token probs for fold/call/raise (both cases, all single tokens in Qwen). Returns probs, entropy, confidence = 1 - H/ln3, hidden state at layer L, regret, led.
- `run_baseline.py` — runs N hands × 3 agents, prints per-hand + summary.
- Env vars: `MODEL` (default Qwen/Qwen2.5-1.5B-Instruct), `DTYPE` (default fp32 on CPU, bf16 on GPU; bf16 on a CPU without native bf16 is very slow).

## Findings so far
- Qwen2.5-0.5B ignores the cards entirely → use 1.5B (or bigger if a GPU is available).
- 1.5B raw output is heavily biased to "fold" regardless of option order or explicit rules.
- Fix = **calibration**: `Brain.calibrate()` averages log p(action) over 20 separate hands with the NEUTRAL persona, subtracts it from every decision. Applied identically to all agents so personalities are preserved. Neutral agent: 3/20 correct raw → 23/30 calibrated on held-out hands.
- Baseline (30 hands, no steering): rock 30 fold, maniac 30 raise, wobbler mixed with confidence ≈ 0.03 (= the jittering one). Personas may be too strong for the knob to move; soften them if step 2 shows that.
- ~0.7 s per decision on a bf16-capable CPU; laptop CPUs need fp32 (bf16 there is several times slower).

## Next steps
1. **Risk vector (make-or-break).** Contrastive prompt pairs (reckless vs cautious persona, same spot), ~50 pairs. Mean hidden state at layer L (last token) for each side; `risk_vec = mean(risky) - mean(safe)`. Save to `risk_vec.pt` (gitignored).
2. **Sweep.** Wobbler agent, c in [-4, -2, -1, 0, 1, 2, 4] on ~20 hands. Expect raise rate ↑ and fold rate ↓ monotonically with c. Find MAX_STRENGTH just below where output breaks (prob mass on actions collapses / gibberish). Also try layers L±4.
3. **Readout.** servo_angle from `hidden @ risk_unit`; jitter from entropy. Check projection correlates with c and with persona.
4. **Live loop.** Agents decide continuously; coef read from shared state every forward pass; keyboard/slider input first.
5. **Pi bridge.** WebSocket/MQTT JSON. Laptop→Pi `{"agent":1,"angle":0-180,"jitter":0-1,"led":true}`; Pi→laptop `{"agent":1,"knob":-1..1}`. Pots need ADC (MCP3008) or Arduino over serial. Knob mapping: raw 0–1023 → k=(raw-511.5)/511.5, dead zone |k|<0.05, c=k*MAX_STRENGTH.
6. Scale beyond poker (any prompt) once the loop works.

## Fallback
If steering doesn't visibly change behaviour, use temperature + logit bias on "raise"/"fold" driven by the knob. Still live, less impressive.

## Working style
Shadman wants concise, direct answers, no filler; push back honestly. Explain new ML concepts briefly when introduced (he's a first-year CS student, strong maths).
