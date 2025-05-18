#!/usr/bin/env python
"""
Evaluate many LSQ heads in one forward pass.
Works with MoE models by letting Accelerate shard across all visible GPUs.
"""

import argparse, torch, tqdm, os, re, math
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--data",  required=True)
p.add_argument("--heads", required=True, help="directory with h*.pt")
p.add_argument("--seq", type=int, default=256)
p.add_argument("--batch_size", type=int, default=128) # Default from patch, adjust as needed
p.add_argument("--rows", default="train[50000:55000]")
p.add_argument("--layer_jump", type=int, default=4)
args = p.parse_args()

# check what layers are available in the heads directory
from pathlib import Path
layers_available = [int(f.stem.split("h")[1]) for f in Path(args.heads).glob("h*.pt")]
layers_to_eval = [l for l in layers_available]
# sort in place
layers_to_eval.sort()

# --- tokenizer ---
print(f"Loading tokenizer for {args.model}...")
tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
tok.pad_token = tok.eos_token if tok.pad_token is None else tok.pad_token

# --- model : shard across all visible GPUs ---
print(f"Loading model {args.model} and sharding across available GPUs...")
# Determine number of available GPUs to set max_memory accurately
num_gpus = torch.cuda.device_count()
print(f"Found {num_gpus} GPUs available.")
max_mem = {i: "75GiB" for i in range(num_gpus)}        # ← use INT keys

model = AutoModelForCausalLM.from_pretrained(
    args.model,
    torch_dtype=torch.bfloat16,
    device_map="balanced",
    max_memory=max_mem,
    low_cpu_mem_usage=True,
    attn_implementation="flash_attention_2",
).eval()

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

# --- dataset ---
print(f"Loading dataset {args.data} split {args.rows}...")
ds = load_dataset("json", data_files=args.data, split=args.rows)
prompts = []
for row in ds:
    if row.get("conversations") and isinstance(row["conversations"], list) and len(row["conversations"]) > 0 and \
       isinstance(row["conversations"][0], dict) and "value" in row["conversations"][0] and \
       row["conversations"][0]["value"]:
        prompts.append(row["conversations"][0]["value"])
    else:
        print(f"Warning: Skipping a row due to invalid/empty conversation data: {row}")

hits = {l: 0 for l in Ws.keys()}     # correct predictions per layer
tots = {l: 0 for l in Ws.keys()}     # token count per layer

# --- evaluation ---
print("Starting evaluation...")
with torch.inference_mode():
    for i in tqdm.tqdm(range(0, len(prompts), args.batch_size), desc="Evaluating Batches"):
        batch_prompts = prompts[i:i+args.batch_size]
        embed_dev = next(iter(model.model.embed_tokens.parameters())).device
        ids = tok(batch_prompts, padding="max_length", truncation=True,
                  max_length=args.seq, return_tensors="pt").to(embed_dev)
        attention_mask = ids["attention_mask"]
        input_ids = ids["input_ids"]
        
        # last_idx is initially on embed_dev
        last_idx_on_embed_dev = attention_mask.sum(dim=1) - 1 

        out = model(**ids, use_cache=False, output_hidden_states=True)
        full_hidden_states = out.hidden_states 

        # --- Gold Labels ---
        logits_device = out.logits.device
        # Move/create indexing tensors on the device of out.logits
        batch_indices_for_logits = torch.arange(input_ids.size(0), device=logits_device)
        last_idx_for_logits = last_idx_on_embed_dev.to(logits_device)
        gold_labels = out.logits[batch_indices_for_logits, last_idx_for_logits].argmax(dim=-1)

        # ------------- build pred labels from early exit + lsq head -----------------
        for l_idx in Ws.keys():
            current_layer_all_hidden_states = full_hidden_states[l_idx + 1]
            hidden_state_device = current_layer_all_hidden_states.device

            # Move/create indexing tensors on the device of the current hidden state
            batch_indices_for_hidden = torch.arange(input_ids.size(0), device=hidden_state_device)
            last_idx_for_hidden = last_idx_on_embed_dev.to(hidden_state_device)
            
            layer_hidden = current_layer_all_hidden_states[batch_indices_for_hidden, last_idx_for_hidden]
            
            lsq = Ws[l_idx] 
            if lsq.device != layer_hidden.device or lsq.dtype != layer_hidden.dtype:
                lsq = lsq.to(layer_hidden.device, dtype=layer_hidden.dtype, non_blocking=True)
            
            lsq_logits = layer_hidden @ lsq.T 
            pred_labels = lsq_logits.argmax(dim=-1)
            
            # ------------- compute accuracy -----------------
            gold_labels_for_comp = gold_labels.to(pred_labels.device)
            
            non_eos_mask = (gold_labels_for_comp != tok.eos_token_id)
            hits[l_idx] += ((gold_labels_for_comp == pred_labels) & non_eos_mask).sum().item()
            tots[l_idx] += non_eos_mask.sum().item()


print("\n--- LSQ Verifier Accuracy (vs. Main Model's Argmax) ---")
for l_idx in Ws.keys():
    if tots[l_idx] > 0:
        acc = hits[l_idx] / tots[l_idx]
        # PPL for LSQ heads is removed
        print(f"L{l_idx:02d} verifier_acc:{acc:6.2%}  tokens_compared:{tots[l_idx]}")
    else:
        print(f"L{l_idx:02d} verifier_acc: N/A (no valid comparisons or head not loaded)")