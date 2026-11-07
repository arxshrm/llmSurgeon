"""
Agents = one shared LLM + a personality (system prompt) + a steering coefficient.

Decision is read in ONE forward pass: we look at the probabilities of the first answer
token being "fold" / "call" / "raise" (no text generation needed).

The steering hook is already wired in. With risk_vec = None or coef = 0 it does nothing,
so this file is the no-steering baseline. Step 2 fills in risk_vec.
"""

import math
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from poker import ACTIONS, Spot

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"   # 0.5B ignores the cards; 1.5B + calibration works. Override with MODEL=...

PERSONAS = {
    "rock":    "You are a tight, cautious poker player. You hate losing chips and only put money in with very strong hands.",
    "maniac":  "You are a loose, aggressive poker player. You love pressure and going all-in, and you rarely fold.",
    "wobbler": "You are a poker player.",
}
INSTRUCTION = "Reply with exactly one word: fold, call, or raise."


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Brain:
    """The shared model. Owns the steering hook."""

    def __init__(self, model_name=None, layer=None, dtype=None):
        import os
        model_name = model_name or os.environ.get("MODEL", DEFAULT_MODEL)
        self.device = pick_device()
        if dtype is None:
            # fp32 on CPU (most laptop CPUs have no fast bf16), bf16 on GPU. Override with DTYPE=fp32|bf16
            default = "fp32" if self.device == "cpu" else "bf16"
            dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}[os.environ.get("DTYPE", default)]
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype).to(self.device).eval()

        n_layers = self.model.config.num_hidden_layers
        self.layer = layer if layer is not None else n_layers // 2   # middle layer is usually best for steering

        # steering state, read by the hook on EVERY forward pass
        self.risk_vec = None      # tensor [hidden_size], set in step 2
        self.coef = 0.0           # set by the knob

        # action token ids (lowercase + capitalised, no leading space: first token of the reply)
        self.action_ids = {
            a: [self.tok.encode(v, add_special_tokens=False)[0] for v in (a, a.capitalize())]
            for a in ACTIONS
        }

        # calibration: the model has a strong built-in bias (e.g. always "fold").
        # bias[a] = average log p(a) over neutral-persona hands; subtracted from every decision.
        self.bias = {a: 0.0 for a in ACTIONS}

        self.model.model.layers[self.layer].register_forward_hook(self._hook)

    def calibrate(self, spots, persona=None):
        """Run once at startup with the NEUTRAL persona and no steering.
        Same correction is applied to every agent, so personalities still differ."""
        persona = persona or PERSONAS["wobbler"]
        saved, self.coef = self.coef, 0.0
        self.bias = {a: 0.0 for a in ACTIONS}
        logs = {a: [] for a in ACTIONS}
        for s in spots:
            probs, _, _ = self.action_probs(persona, s)
            for a in ACTIONS:
                logs[a].append(math.log(probs[a]))
        self.bias = {a: sum(v) / len(v) for a, v in logs.items()}
        self.coef = saved
        return self.bias

    def _hook(self, module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        if self.risk_vec is not None and self.coef != 0.0:
            hidden = hidden + self.coef * self.risk_vec.to(hidden.dtype)   # h_new = h + c * v
        if isinstance(output, tuple):
            return (hidden,) + tuple(output[1:])
        return hidden

    def prompt_ids(self, persona: str, spot: Spot):
        msgs = [
            {"role": "system", "content": f"{persona} {INSTRUCTION}"},
            {"role": "user", "content": spot.describe()},
        ]
        text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return self.tok(text, return_tensors="pt").to(self.device)

    @torch.no_grad()
    def action_probs(self, persona: str, spot: Spot):
        ids = self.prompt_ids(persona, spot)
        out = self.model(**ids, output_hidden_states=True)
        p = torch.softmax(out.logits[0, -1].float(), dim=-1)
        raw = {a: sum(p[i].item() for i in self.action_ids[a]) for a in ACTIONS}
        mass = sum(raw.values())                       # sanity: should be close to 1
        # calibrated scores -> softmax over the 3 actions
        score = {a: math.log(raw[a] + 1e-12) - self.bias[a] for a in ACTIONS}
        m = max(score.values())
        z = sum(math.exp(v - m) for v in score.values())
        probs = {a: math.exp(score[a] - m) / z for a in ACTIONS}
        # hidden state at the steering layer, last token (+1 because index 0 = embeddings)
        hidden = out.hidden_states[self.layer + 1][0, -1].float().cpu()
        return probs, mass, hidden


def entropy(probs: dict) -> float:
    return -sum(p * math.log(p) for p in probs.values() if p > 0)


class Agent:
    def __init__(self, name: str, brain: Brain, coef: float = 0.0):
        self.name = name
        self.persona = PERSONAS[name]
        self.brain = brain
        self.coef = coef          # this agent's knob position

    def decide(self, spot: Spot) -> dict:
        self.brain.coef = self.coef          # load this agent's knob into the shared hook
        probs, mass, hidden = self.brain.action_probs(self.persona, spot)
        action = max(probs, key=probs.get)
        H = entropy(probs)
        return {
            "agent": self.name,
            "probs": probs,
            "action": action,
            "entropy": H,
            "confidence": 1 - H / math.log(len(ACTIONS)),   # 1 = certain, 0 = totally torn
            "mass": mass,
            "hidden": hidden,
            "best": spot.best_action(),
            "regret": spot.regret(action),
            "led": action == spot.best_action(),
        }
