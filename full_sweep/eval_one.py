#!/usr/bin/env python
"""
Evaluate a single LSQ head against the full model on 5k prompts.
Prints accept-rate.
"""

import argparse, torch, tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

# ---------- CLI ----------
p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--data", required=True)
p.add_argument("--layer", type=int, required=True)
p.add_argument("--head", required=True, help="path to W for this layer")
p.add_argument("--seq", type=int, default=256)
p.add_argument("--rows", default="train[50000:55000]", help="HF split spec for eval set")
args = p.parse_args()

# ---------- load ----------
tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
model = AutoModelForCausalLM.from_pretrained(
    args.model,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="flash_attention_2",
).eval()
W = torch.load(args.head, map_location="cuda")  # (V,d)

test_ds = load_dataset("json", data_files=args.data, split=args.rows)

hits = tot = 0
with torch.no_grad():
    for row in tqdm.tqdm(test_ds, desc=f"L{args.layer}"):
        if not row["conversations"]:
            continue
        ids = tok(
            row["conversations"][0]["value"],
            return_tensors="pt",
            truncation=True,
            max_length=args.seq,
        ).to("cuda")

        out = model(**ids, use_cache=False, output_hidden_states=True)
        h = out.hidden_states[args.layer + 1][:, -1, :]  # (1,d)
        pred = (h @ W.T).argmax(-1)
        gold = out.logits[:, -1, :].argmax(-1)
        hits += (pred == gold).item()
        tot += 1

print(f"Layer {args.layer} accept-rate: {hits/tot:.2%}")
