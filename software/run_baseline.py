"""
Step 1: baseline, no steering.
Each agent plays the same N spots. Prints per-hand decisions + a summary table.

    python run_baseline.py            # 20 hands
    python run_baseline.py 50         # 50 hands
"""

import sys, random, collections
from poker import deal_spot, ACTIONS
from agent import Brain, Agent, PERSONAS

N = int(sys.argv[1]) if len(sys.argv) > 1 else 20

brain = Brain()
print(f"device={brain.device}  steering layer={brain.layer}/{brain.model.config.num_hidden_layers}\n")
agents = [Agent(name, brain) for name in PERSONAS]

calib = [deal_spot(random.Random(10_000 + i)) for i in range(20)]   # separate hands, never scored
print(f"calibrating on {len(calib)} hands (no output until done)...", flush=True)
import time; _t = time.time()
bias = brain.calibrate(calib)
print("bias:", {a: round(v, 2) for a, v in bias.items()},
      f"| {(time.time() - _t) / len(calib):.1f}s per decision\n", flush=True)

rng = random.Random(42)
spots = [deal_spot(rng) for _ in range(N)]

stats = {a.name: {"acts": collections.Counter(), "regret": 0.0, "led": 0, "conf": 0.0, "mass": 0.0} for a in agents}

for i, spot in enumerate(spots, 1):
    print(f"--- hand {i}/{N}: eq={spot.equity:.2f} pot_odds={spot.pot_odds:.2f} best={spot.best_action()}")
    for ag in agents:
        r = ag.decide(spot)
        s = stats[ag.name]
        s["acts"][r["action"]] += 1
        s["regret"] += r["regret"]
        s["led"] += r["led"]
        s["conf"] += r["confidence"]
        s["mass"] += r["mass"]
        p = "  ".join(f"{a}={r['probs'][a]:.2f}" for a in ACTIONS)
        print(f"  {ag.name:8s} {p}  -> {r['action']:5s} conf={r['confidence']:.2f} "
              f"{'GREEN' if r['led'] else '     '} regret={r['regret']:.0f}")

print("\n=== summary ===")
print(f"{'agent':8s} {'fold':>5s} {'call':>5s} {'raise':>5s} {'LED%':>5s} {'avg regret':>11s} {'avg conf':>9s} {'mass':>5s}")
for name, s in stats.items():
    a = s["acts"]
    print(f"{name:8s} {a['fold']:5d} {a['call']:5d} {a['raise']:5d} {100*s['led']/N:5.0f} "
          f"{s['regret']/N:11.1f} {s['conf']/N:9.2f} {s['mass']/N:5.2f}")
best = collections.Counter(sp.best_action() for sp in spots)
print(f"{'OPTIMAL':8s} {best['fold']:5d} {best['call']:5d} {best['raise']:5d}")
