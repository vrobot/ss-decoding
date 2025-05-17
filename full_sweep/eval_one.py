#!/usr/bin/env python
"""
Evaluate a single LSQ head against the full model on 5k prompts.
Prints accept-rate.
"""

import argparse
import torch
import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import os

# Suggested: For PyTorch 2.0+ to allow TF32 on Ampere/Hopper GPUs
# if torch.__version__ >= "2.0":
#     torch.set_float32_matmul_precision("high")

# ---------- CLI ----------
p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--data", required=True)
p.add_argument("--layer", type=int, required=True)
p.add_argument("--head", required=True, help="path to W for this layer")
p.add_argument("--seq", type=int, default=256)
p.add_argument("--rows", default="train[50000:55000]", help="HF split spec for eval set")
p.add_argument("--batch_size", type=int, default=64, help="Batch size for evaluation")
args = p.parse_args()

# ---------- load ----------
print(f"Loading tokenizer for {args.model}...")
tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

print(f"Loading model {args.model} for layer {args.layer} evaluation...")
# Force model to the single visible GPU (which is cuda:0 due to CUDA_VISIBLE_DEVICES)
model_loading_device_map = {"": 0} # Maps to the GPU made visible by CUDA_VISIBLE_DEVICES

model = AutoModelForCausalLM.from_pretrained(
    args.model,
    torch_dtype=torch.bfloat16,
    device_map=model_loading_device_map,
    low_cpu_mem_usage=False,
    attn_implementation="flash_attention_2",
).eval()

# Suggested: PyTorch 2.0+ compile for speedup
# if torch.__version__ >= "2.0":
#     print(f"Attempting to compile model for layer {args.layer}...")
#     try:
#         # Using fullgraph=False can sometimes be more robust for complex models or certain ops
#         model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
#         print("Model compilation complete.")
#     except Exception as e:
#         print(f"Model compilation failed: {e}. Proceeding without compilation.")

# W is loaded onto the same device as the model will effectively be on.
# Since CUDA_VISIBLE_DEVICES restricts to one GPU, "cuda" here refers to that GPU.
W = torch.load(args.head, map_location="cuda")  # (V,d)
print(f"Loaded LSQ head {args.head} to device: {W.device}")
# model.device might point to 'meta' or a specific device depending on `device_map` internals,
# but operations will happen on the device specified in device_map.
# For single device mapping like {"":0}, model.device should reflect that.
print(f"Model's primary device (post device_map): {model.device}")


print(f"Loading dataset {args.data} split {args.rows}...")
test_ds = load_dataset("json", data_files=args.data, split=args.rows)

# Pre-collect prompts
prompts = []
for row_idx, row in enumerate(test_ds):
    if row["conversations"] and isinstance(row["conversations"], list) and len(row["conversations"]) > 0 and \
       isinstance(row["conversations"][0], dict) and "value" in row["conversations"][0] and \
       row["conversations"][0]["value"]:
        prompts.append(row["conversations"][0]["value"])
    else:
        print(f"Warning: Skipping row index {row_idx} due to invalid/empty conversation data: {row}")

if not prompts:
    print("No valid prompts found in the dataset slice. Exiting.")
    exit()

print(f"Collected {len(prompts)} prompts for evaluation.")

hits = 0
tot = 0

# Use torch.inference_mode() for evaluation
with torch.inference_mode():
    # Batch processing
    for i in tqdm.tqdm(range(0, len(prompts), args.batch_size), desc=f"L{args.layer} Eval (BS: {args.batch_size})"):
        batch_prompts = prompts[i:i + args.batch_size]
        
        ids = tok(
            batch_prompts,
            padding="longest", # Tokenizer already has padding_side="left"
            truncation=True,
            max_length=args.seq,
            return_tensors="pt"
        ).to(model.device) # Ensure input tensors are on the same device as the model

        out = model(**ids, use_cache=True, output_hidden_states=True)
        
        # Ensure h is on the same device as W for the matmul
        # out.hidden_states elements are tuples; the actual tensor is the first element
        hidden_state_tensor = out.hidden_states[args.layer + 1]
        h = hidden_state_tensor[:, -1, :].to(W.device)  # (B,d)
        pred = (h @ W.T).argmax(-1) # pred should be on W.device
        
        # Ensure gold is on the same device as pred for comparison
        # out.logits is the tensor directly
        gold = out.logits[:, -1, :].argmax(-1).to(pred.device)
        
        hits += (pred == gold).sum().item()
        tot += len(batch_prompts)

if tot > 0:
    print(f"Layer {args.layer} accept-rate: {hits/tot:.2%}")
else:
    print(f"Layer {args.layer} accept-rate: N/A (no samples processed)")
