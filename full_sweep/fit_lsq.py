#!/usr/bin/env python
"""
Compute LSQ heads Wₗ  = Yᵀ X (Xᵀ X + λI)⁻¹ for all layers.
Processes layers in batches to fit GPU memory for accumulators.
Expensive matrix ops are done on GPU. Shards are read once per layer batch.
Saves bf16 matrices to out_dir.
"""

import argparse
import glob
import torch
import tqdm
import os
import math

p = argparse.ArgumentParser(
    description="Fit LSQ heads for all layers using batched layer processing on GPU."
)
p.add_argument("--act_dir", required=True, help="Directory where dump_all.py stored *.pt shards")
p.add_argument(
    "--out_dir", required=True, help="Directory to write head files (e.g., h0.pt, h1.pt)"
)
p.add_argument("--lambda_", type=float, default=1e-4, help="Ridge factor for LSQ")
p.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use for computation")
p.add_argument(
    "--layers_per_batch",
    type=int,
    default=24,
    help="Number of layers to process in one batch (adjust based on GPU memory). Accumulators for these layers will be on GPU.",
)
p.add_argument(
    "--acc_dtype",
    choices=["float32", "float16", "bfloat16"],
    default="float16",
    help="Data type for accumulation buffers. Using float16 reduces memory usage but may slightly affect accuracy.",
)
p.add_argument("--layer_jump", type=int, default=4)
args = p.parse_args()

dtype_map = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}
acc_dtype = dtype_map[args.acc_dtype]
print(f"Script called with arguments: {args}")

shards = sorted(glob.glob(f"{args.act_dir}/b*.pt"))
if not shards:
    raise FileNotFoundError(f"No shards found in {args.act_dir} – did dump_all.py run?")

print(f"Found {len(shards)} shards in {args.act_dir}")

# --- Determine device for solving (GPU if available) ---
if not torch.cuda.is_available():
    raise RuntimeError("CUDA not available. This script requires a GPU.")
device = torch.device(f"cuda:{args.gpu_id % torch.cuda.device_count()}")
print(f"Using device: {device} for accumulation and solving.")
torch.cuda.set_device(device)  # Set default CUDA device

# --- Discover layers, d_model, and vocab_size from the first shard ---
print("Inspecting first shard for metadata...")
first_shard_path = shards[0]
meta_blob = torch.load(first_shard_path, map_location="cpu")

all_layer_keys = sorted([k for k in meta_blob.keys() if k.startswith("h")])
if not all_layer_keys:
    raise ValueError("No layer keys (e.g., 'h0', 'h1') found in the first shard.")

if "logits" not in meta_blob:
    raise ValueError(f"Logits not found in first shard: {first_shard_path}")

d_model = meta_blob[all_layer_keys[0]].shape[1]
vocab_size = meta_blob["logits"].shape[1]
num_total_layers = len(all_layer_keys) # CORRECTED: This is the number of layers we have activations for
print(f"Discovered {len(all_layer_keys)} activation files (e.g., {all_layer_keys[:3]}...). d_model={d_model}, vocab_size={vocab_size}")
del meta_blob

os.makedirs(args.out_dir, exist_ok=True)

# The layers_per_batch logic should operate on num_total_layers, which is now correct
num_layer_batches = math.ceil(num_total_layers / args.layers_per_batch)
print(
    f"Processing {num_total_layers} layers in {num_layer_batches} batches of up to {args.layers_per_batch} layers each."
)

for batch_idx in range(num_layer_batches):
    start_idx_for_slicing = batch_idx * args.layers_per_batch # This is an index into all_layer_keys
    end_idx_for_slicing = min(start_idx_for_slicing + args.layers_per_batch, num_total_layers)
    current_batch_layer_keys = all_layer_keys[start_idx_for_slicing:end_idx_for_slicing]

    if not current_batch_layer_keys:
        continue

    print(
        f"\n--- Processing Layer Batch {batch_idx+1}/{num_layer_batches} (Layers {current_batch_layer_keys[0]} to {current_batch_layer_keys[-1]}) ---"
    )

    # --- Initialize accumulators on GPU for the current batch of layers ---
    print(
        f"Initializing GPU accumulators ({args.acc_dtype}) for {len(current_batch_layer_keys)} layers in this batch..."
    )
    XtX_batch_gpu = {k: torch.zeros((d_model,d_model),
                                 dtype=torch.float32,
                                 device=device) for k in current_batch_layer_keys}
    YtX_batch_gpu = {k: torch.zeros((vocab_size,d_model),
                                 dtype=torch.float32,
                                 device=device) for k in current_batch_layer_keys}
    print("GPU accumulators for batch initialized.")

    print(f"Starting accumulation pass over {len(shards)} shards for current layer batch...")
    # --- Accumulate XtX and YtX from all shards for the current batch of layers ---
    for shard_path in tqdm.tqdm(shards, desc=f"Shards (Batch {batch_idx+1})"):
        try:
            blob = torch.load(shard_path, map_location="cpu")

            # Logits are common for all layers in this batch for this shard
            Y_shard_cpu = blob["logits"]
            # Move to GPU with requested dtype and clamp
            current_logits_gpu = Y_shard_cpu.to(device=device, dtype=torch.float32)
            current_logits_gpu.clamp_(-3000.0, 3000.0)  # Clamp large values

            for layer_key in current_batch_layer_keys:
                if layer_key not in blob:
                    print(f"Warning: Layer {layer_key} not found in shard {shard_path}. Skipping.")
                    continue

                X_shard_layer_cpu = blob[layer_key]
                # Move to GPU with requested dtype and clamp
                X_shard_layer_gpu = X_shard_layer_cpu.to(device=device, dtype=torch.float32)
                X_shard_layer_gpu.clamp_(-3000.0, 3000.0)  # Clamp large values

                # Matmuls in the chosen dtype; accumulators share the same dtype
                XtX_batch_gpu[layer_key] += X_shard_layer_gpu.T @ X_shard_layer_gpu
                YtX_batch_gpu[layer_key] += current_logits_gpu.T @ X_shard_layer_gpu

                del X_shard_layer_gpu
                torch.cuda.empty_cache()

            del blob, current_logits_gpu
        except Exception as e:
            print(
                f"Error processing shard {shard_path} for layer batch {batch_idx+1}: {e}. Skipping shard."
            )
            continue
    print("Accumulation pass for current layer batch complete.")

    # --- Solve for W for each layer in the current batch ---
    print(f"Solving for W for layers in batch {batch_idx+1}...")
    for layer_key in tqdm.tqdm(current_batch_layer_keys, desc=f"Solving (Batch {batch_idx+1})"):
        XtX_l_dev = XtX_batch_gpu[layer_key]
        YtX_l_dev = YtX_batch_gpu[layer_key]

        A_l = XtX_l_dev.to(torch.float32) + args.lambda_ * torch.eye(
            d_model, dtype=torch.float32, device=device
        )

        YtX_T_dev = YtX_l_dev.to(torch.float32).T

        # Using torch.linalg.lstsq for robustness against singular matrices
        try:
            # Check for NaNs/Infs in A_l BEFORE the solve
            if not torch.isfinite(A_l).all():
                print(
                    f"Warning: NaNs or Infs detected in A_l for layer {layer_key} BEFORE lstsq. XtX condition: {torch.linalg.cond(XtX_l_dev) if torch.isfinite(XtX_l_dev).all() else 'N/A or Inf/NaN in XtX'}"
                )

            # print(f"Attempting solve for {layer_key} with lambda={args.lambda_}. A_l condition number: {torch.linalg.cond(A_l) if A_l.shape[0] == A_l.shape[1] and torch.isfinite(A_l).all() else 'N/A or Inf/NaN'}")

            lstsq_solution = torch.linalg.lstsq(A_l, YtX_T_dev)
            W_T_dev = lstsq_solution.solution
            W_l_dev = W_T_dev.T
            if torch.isnan(W_l_dev).any() or torch.isinf(W_l_dev).any():
                print(
                    f"Warning: NaNs or Infs detected in W_l_dev for layer {layer_key} after lstsq. This might indicate severe issues."
                )
        except RuntimeError as e:
            print(f"torch.linalg.lstsq failed for layer {layer_key}: {e}")
            print(f"Skipping save for layer {layer_key} due to lstsq error.")
            # Optionally, save a zero matrix or skip saving
            # W_l_dev = torch.zeros_like(YtX_l_dev.T, dtype=torch.bfloat16, device='cpu') # Example: save zeros
            continue  # Skip saving this problematic layer

        output_filename = f"{layer_key}.pt"
        output_path = os.path.join(args.out_dir, output_filename)

        try:
            torch.save(W_l_dev.to(torch.bfloat16).cpu(), output_path)
            # print(f"Saved: {output_path}") # tqdm provides progress
        except Exception as e:
            print(f"Error saving {output_path}: {e}")

    print(f"Finished solving for layer batch {batch_idx+1}.")
    del XtX_batch_gpu, YtX_batch_gpu  # Free GPU memory for accumulators of this batch
    torch.cuda.empty_cache()


print("\nAll LSQ heads computed and saved.")
