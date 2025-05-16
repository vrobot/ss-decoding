#!/usr/bin/env python
"""
Stream N prompts through a model, dump the last-token hidden state for
every decoder layer plus the final logits, one shard per batch.

Output:  acts_dir/b00000.pt, b00001.pt …  (dict{ h0,h1,…,logits })
"""

import argparse, os, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm

# ---------------- CLI ----------------
p = argparse.ArgumentParser()
p.add_argument("--model", required=True,
               help="HF model id or local path")
p.add_argument("--data", required=True,
               help="ShareGPT JSON file")
p.add_argument("--n", type=int, default=50_000,
               help="number of prompts to stream")
p.add_argument("--batch", type=int, default=8,
               help="batch size per forward pass")
p.add_argument("--seq", type=int, default=256,
               help="truncate/pad length")
p.add_argument("--out", default="acts",
               help="output directory")
args = p.parse_args()

N, B, SEQ = args.n, args.batch, args.seq

# --------------- model --------------
tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="flash_attention_2"
            ).eval()

layers = []
cache  = {}                       # key -> tensor (B, S, d)

def make_hook(idx):
    key = f"h{idx}"
    layers.append(key)
    def _hook(_,__,out):
        # out = (hidden, present_kv)
        cache[key] = out[0][:,-1,:].bfloat16().cpu()
    return _hook

for i, blk in enumerate(model.model.layers):
    blk.register_forward_hook(make_hook(i))

# --------------- data ---------------
ds = load_dataset("json", data_files=args.data,
                  split=f"train[:{N}]")

os.makedirs(args.out, exist_ok=True)

# ------------- forward --------------
with torch.no_grad():
    for batch_idx in tqdm(range(0, N, B)):
        prompts = []
        for j in range(batch_idx, min(batch_idx + B, N)):
            row = ds[j]
            if row["conversations"]:
                prompts.append(row["conversations"][0]["value"])
        if not prompts:
            continue

        ids = tok(prompts, return_tensors="pt",
                  padding="max_length", truncation=True,
                  max_length=SEQ).to("cuda")
        cache.clear()
        out = model(**ids, use_cache=False)
        shard = {k: cache[k] for k in layers}   # take last tok
        shard["logits"] = out.logits[:, -1, :].cpu()
        torch.save(shard, f"{args.out}/b{batch_idx//B:05d}.pt")

