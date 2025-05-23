#!/usr/bin/env python
"""
Stream N prompts through a model, dump the *last-real-token* hidden state for
every decoder layer plus the final logits, one shard per batch.

Output: acts_dir/b00000.pt, b00001.pt …   with keys  {h0, h1, …, logits}
"""

import argparse, os, torch, json
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets    import load_dataset
from tqdm        import tqdm

# ---------- CLI ----------
p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--data",  required=True)        # ShareGPT JSON
p.add_argument("--n",     type=int, default=50_000)
p.add_argument("--batch", type=int, default=256)
p.add_argument("--seq",   type=int, default=256)
p.add_argument("--out",   default="acts_output")
p.add_argument("--layer_jump", type=int, default=4)
args = p.parse_args()

N, B, SEQ = args.n, args.batch, args.seq
os.makedirs(args.out, exist_ok=True)

# ---------- model ----------
tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
tok.pad_token = tok.eos_token if tok.pad_token is None else tok.pad_token

model = AutoModelForCausalLM.from_pretrained(
    args.model,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="flash_attention_2"
).eval()

# pre-register a tiny hook for every block
cache = {}                                  # filled on-the-fly by hooks
last_idx = None                             # will be set each batch

def make_hook(i):
    key = f"h{i}"
    def _hook(_, __, out):
        # out = (hidden, present_kv)
        h = out[0]                          # (B, S, d) - this tensor could be on any GPU
        current_h_device = h.device
        
        # Ensure last_idx is on the same device as h for indexing
        # The `last_idx` variable is captured from the outer scope of the batch loop
        last_idx_on_h_device = last_idx.to(current_h_device)
        
        batch_indices_on_h_device = torch.arange(h.size(0), device=current_h_device)
        
        cache[key] = h[batch_indices_on_h_device, last_idx_on_h_device].bfloat16().cpu()
    return _hook

for i, blk in enumerate(model.model.layers):
    blk.register_forward_hook(make_hook(i))

# ---------- data ----------
ds = load_dataset("json", data_files=args.data,
                  split=f"train[:{N}]")

# ---------- forward loop ----------
with torch.no_grad():
    for b_ofs in tqdm(range(0, N, B)):
        prompts = []
        for j in range(b_ofs, min(b_ofs + B, N)):
            row = ds[j]
            if row.get("conversations"):
                prompts.append(row["conversations"][0]["value"])
        if not prompts:
            continue

        dev = next(iter(model.model.embed_tokens.parameters())).device
        ids = tok(prompts, return_tensors="pt",
                  padding="max_length", truncation=True,
                  max_length=SEQ).to(dev)

        # make last_idx visible to the hooks
        last_idx = ids["attention_mask"].sum(1) - 1      # (B,)

        out = model(**ids,
                    use_cache=False,
                    output_hidden_states=True)            # tuple len = L+1

        shard = {}
        for i in range(0, len(model.model.layers), args.layer_jump):
            key = f"h{i}"
            if i + 1 < len(out.hidden_states):
                h = out.hidden_states[i+1]  # (B,SEQ,d)
                h_last = h[torch.arange(h.size(0), device=h.device), last_idx]
                shard[key] = h_last.bfloat16().cpu()
            else:
                print(f"Warning: Layer index {i} (from layer_jump) is out of bounds for hidden_states. Skipping {key}.")

        shard["logits"] = out.logits[
            torch.arange(out.logits.size(0), device=dev), last_idx
        ].cpu()

        torch.save(shard, f"{args.out}/b{b_ofs//B:05d}.pt")
