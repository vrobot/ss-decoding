#!/usr/bin/env python
"""
Evaluate many LSQ heads in one forward pass.
Works with MoE models by letting Accelerate shard across all visible GPUs.
"""

import argparse, torch, tqdm, os, re, math
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

# Suggested: For PyTorch 2.0+ to allow TF32 on Ampere/Hopper GPUs
# if torch.__version__ >= "2.0":
#     torch.set_float32_matmul_precision("high")

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--data",  required=True)
p.add_argument("--heads", required=True, help="directory with h*.pt")
p.add_argument("--layers", default="0:48:4",
               help="start:stop:step layer spec (python slice syntax)")
p.add_argument("--seq", type=int, default=256)
p.add_argument("--batch_size", type=int, default=128) # Default from patch, adjust as needed
p.add_argument("--rows", default="train[50000:55000]")
args = p.parse_args()

# --- parse layer slice ---
m = re.match(r"(\d+):(\d+):(\d+)", args.layers)
if not m:
    raise ValueError(f"Invalid layer spec: {args.layers}. Expected format 'start:stop:step'")
start, stop, step = map(int, m.groups())
layers_to_eval = list(range(start, stop, step))

print(f"🔹 Layers to evaluate: {layers_to_eval}")
if not layers_to_eval:
    print("No layers selected for evaluation. Exiting.")
    exit()

# --- tokenizer ---
print(f"Loading tokenizer for {args.model}...")
tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
tok.pad_token = tok.eos_token if tok.pad_token is None else tok.pad_token

# --- model : shard across all visible GPUs ---
print(f"Loading model {args.model} and sharding across available GPUs...")
# Determine number of available GPUs to set max_memory accurately
num_gpus = torch.cuda.device_count()
print(f"Found {num_gpus} GPUs available.")
# 1. string keys
max_mem = {i: "38GiB" for i in range(num_gpus)}        # ← use INT keys

model = AutoModelForCausalLM.from_pretrained(
    args.model,
    torch_dtype=torch.bfloat16,
    device_map="balanced",
    max_memory=max_mem,
    low_cpu_mem_usage=True,          # stream from disk, uses little RAM
    attn_implementation="flash_attention_2", # Ensure this is compatible
).eval()

# # Suggested: PyTorch 2.0+ compile for speedup
# if torch.__version__ >= "2.0":
#     print("Attempting to compile model...")
#     try:
#         model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
#         print("Model compilation complete.")
#     except Exception as e:

# --- load LSQ heads (always fits on gpu:0) ---
print(f"Loading LSQ heads from {args.heads} to cuda:0...")
Ws = {}
for l_idx in layers_to_eval:
    head_path = os.path.join(args.heads, f"h{l_idx}.pt")
    if not os.path.exists(head_path):
        print(f"Warning: Head file {head_path} not found for layer {l_idx}. Skipping this layer.")
        continue
    # 3. keep heads on CPU
    Ws[l_idx] = torch.load(head_path, map_location="cpu")
print(f"Loaded {len(Ws)} LSQ heads.")

if not Ws:
    print("No LSQ heads were loaded. Exiting.")
    exit()

# --- dataset ---
print(f"Loading dataset {args.data} split {args.rows}...")
ds = load_dataset("json", data_files=args.data, split=args.rows)
prompts = []
for row in ds:
    if row.get("conversations") and isinstance(row["conversations"], list) and len(row["conversations"]) > 0 and \
       isinstance(row["conversations"][0], dict) and "value" in row["conversations"][0] and \
       row["conversations"][0]["value"]:
        prompts.append(row["conversations"][0]["value"])
    # else:
    #     print(f"Warning: Skipping a row due to invalid/empty conversation data: {row}")


if not prompts:
    print("No valid prompts found in the dataset slice. Exiting.")
    exit()
print(f"Collected {len(prompts)} prompts for evaluation.")


hits = {l: 0 for l in Ws.keys()}     # correct predictions per layer
tots = {l: 0 for l in Ws.keys()}     # token count per layer
nll_sum = {l: 0.0 for l in Ws.keys()} # accumulated NLL per layer
total_nll = 0.0
total_tokens = 0
printed_example = False  # sanity print only once

print("Starting evaluation...")
with torch.inference_mode():
    for i in tqdm.tqdm(range(0, len(prompts), args.batch_size), desc="Evaluating Batches"):
        batch_prompts = prompts[i:i+args.batch_size]
        # 2. send inputs to whichever GPU holds the embedding layer
        embed_dev = next(iter(model.model.embed_tokens.parameters())).device
        ids = tok(batch_prompts, padding="max_length", truncation=True,
                  max_length=args.seq, return_tensors="pt").to(embed_dev)

        # ------------- build gold labels from source text -----------------
        input_ids = ids["input_ids"]              # (B, S)
        labels = input_ids.clone()
        labels[labels == tok.pad_token_id] = -100

        gold = []
        for b in range(labels.size(0)):
            last = (ids["attention_mask"][b] == 1).nonzero(as_tuple=False).max()
            if last + 1 < labels.size(1):
                gold.append(labels[b, last + 1])
            else:
                gold.append(tok.eos_token_id)
        gold = torch.tensor(gold, device=labels.device)

        out = model(**ids, use_cache=True, output_hidden_states=True)

        lm_loss = F.cross_entropy(
            out.logits.float().view(-1, out.logits.size(-1)),
            labels.view(-1),
            ignore_index=-100,
            reduction="sum",
        )
        total_nll += lm_loss.item()
        total_tokens += (labels != -100).sum().item()

        attn_mask = ids['attention_mask']          # (B, SEQ)
        last_idx = attn_mask.sum(dim=1) - 1       # (B,)

        keep = gold != tok.eos_token_id          # ignore EOS tokens

        for l_idx in Ws.keys(): # Iterate over successfully loaded heads
            hidden = out.hidden_states[l_idx+1]                     # (B,S,d) on dev_L
            h_last = hidden[
                torch.arange(hidden.size(0), device=hidden.device), last_idx
            ]                                                      # stays on dev_L

            # 1. move (once!) the LSQ head to **that same device** + dtype
            if Ws[l_idx].device != hidden.device or Ws[l_idx].dtype != hidden.dtype:
                Ws[l_idx] = Ws[l_idx].to(hidden.device,
                                          dtype=hidden.dtype,  # bfloat16
                                          non_blocking=True)

            W = Ws[l_idx]
            logits_h = h_last @ W.T                    # (B, V)  bfloat16
            if keep.any():
                pred = logits_h.argmax(-1)             # still on bfloat16
                hits[l_idx] += (pred[keep] == gold[keep].to(pred.device)).sum().item()
                tots[l_idx] += keep.sum().item()
                loss = F.cross_entropy(logits_h[keep].float(),
                                       gold[keep].to(logits_h.device),
                                       reduction='sum')
                nll_sum[l_idx] += loss.item()
                sample_i = torch.randint(len(batch_prompts), (1,)).item()
                if not printed_example and (pred[sample_i] != gold[sample_i]):
                    print("\nPrompt:", batch_prompts[sample_i][-200:])
                    print("Gold token:", tok.decode([gold[sample_i]]).strip())
                    top_k = logits_h[sample_i].float().softmax(-1).topk(5)
                    print("LSQ top-5:",
                          [(tok.decode([idx]), float(p))
                           for p, idx in zip(top_k.values, top_k.indices)])
                    printed_example = True

print("\n--- Accuracy vs ground-truth next token ---")
for l_idx in Ws.keys():
    if tots[l_idx] > 0:
        acc = hits[l_idx] / tots[l_idx]
        ppl = math.exp(float(nll_sum[l_idx]) / tots[l_idx])
        print(f"L{l_idx:02d} acc:{acc:5.2%}  ppl:{ppl:6.2f}  tokens:{tots[l_idx]}")
    else:
        print(f"Layer {l_idx:02d} acc: N/A (no samples processed or head not loaded)")

if total_tokens > 0:
    full_ppl = math.exp(total_nll / total_tokens)
    print(f"Unconditional PPL over prefix: {full_ppl:.2f}")
