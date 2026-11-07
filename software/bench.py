"""
Run this FIRST on a new machine. It answers the one question that decides the architecture:
can agents generate a line of text live, or only vote?

    python bench.py              # fp32 (right choice on most laptop CPUs)
    python bench.py bf16         # compare; bf16 is pathologically slow without native support
    python bench.py fp32 bf16    # both

Read the output like this:
  vote round   < 3s   -> vote-based debate works (~10+ rounds in 30s). Design C/B viable.
  tok/s        > 4    -> spotlight agent can narrate live (design B).
  tok/s        > 10   -> all three agents can narrate (design A).
  tok/s        < 1    -> text is off the table; templated readout only (design C).
"""
import sys, time, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

NAME = "Qwen/Qwen2.5-1.5B-Instruct"
DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16}
want = [a for a in sys.argv[1:] if a in DTYPES] or ["fp32"]

tok = AutoTokenizer.from_pretrained(NAME)
msgs = [
    {"role": "system", "content": "You are an on-call engineer. One short line, then your choice."},
    {"role": "user", "content": "Alert: checkout-api error rate 12%, 3400 users hit, "
                                "deploy 4 min ago. Real-incident chance 71%. "
                                "Options: wait, page, restart."},
]
ids = tok(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True),
          return_tensors="pt")

print(f"threads={torch.get_num_threads()}  prompt_tokens={ids.input_ids.shape[1]}", flush=True)

for label in want:
    m = AutoModelForCausalLM.from_pretrained(NAME, dtype=DTYPES[label]).eval()
    with torch.no_grad():
        t = time.time(); m(**ids); cold = time.time() - t      # includes one-time setup
        t = time.time(); m(**ids); warm = time.time() - t      # the number that matters
        print(f"\n[{label}] cold first pass = {cold:.2f}s   (warm up before demoing!)", flush=True)
        print(f"[{label}] warm prefill    = {warm:.2f}s  -> 3-agent VOTE round = {3*warm:.1f}s", flush=True)
        t = time.time()
        out = m.generate(**ids, max_new_tokens=10, do_sample=False, pad_token_id=tok.eos_token_id)
        gen = time.time() - t
    n = out.shape[1] - ids.input_ids.shape[1]
    tps = n / gen
    print(f"[{label}] generation      = {n} tok in {gen:.1f}s = {tps:.1f} tok/s")
    print(f"[{label}] 12-tok line: 1 agent {warm + 12/tps:5.1f}s | 3 agents {3*(warm + 12/tps):5.1f}s")
    print(f"[{label}] text: {tok.decode(out[0, ids.input_ids.shape[1]:], skip_special_tokens=True)!r}")
    del m
