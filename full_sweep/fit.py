#!/usr/bin/env python
"""Fit LSQ heads using ridge regression: W = Y^T X (X^T X + λI)^-1"""

import argparse
import torch
import glob
import os
from tqdm import tqdm

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--act_dir", required=True, help="Directory with activation files")
    p.add_argument("--out_dir", required=True, help="Directory to save LSQ heads")
    p.add_argument("--lambda_", type=float, default=1e-4, help="Ridge regularization")
    args = p.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Find all activation files
    files = sorted(glob.glob(f"{args.act_dir}/*.pt"))
    if not files:
        raise FileNotFoundError(f"No .pt files found in {args.act_dir}")
    
    print(f"Found {len(files)} activation files")
    
    # Get dimensions from first file
    first_batch = torch.load(files[0], map_location="cpu")
    layer_keys = sorted([k for k in first_batch.keys() if k.startswith("h")])
    d_model = first_batch[layer_keys[0]].shape[1]
    vocab_size = first_batch["logits"].shape[1]
    
    print(f"Processing {len(layer_keys)} layers, d_model={d_model}, vocab_size={vocab_size}")
    
    # Process each layer
    for layer_key in tqdm(layer_keys, desc="Fitting LSQ heads"):
        XtX = torch.zeros(d_model, d_model, dtype=torch.float32)
        YtX = torch.zeros(vocab_size, d_model, dtype=torch.float32)
        
        # Accumulate over all batches
        for file_path in files:
            batch = torch.load(file_path, map_location="cpu")
            if layer_key not in batch:
                continue
                
            X = batch[layer_key].float()  # (B, d_model)
            Y = batch["logits"].float()   # (B, vocab_size)
            
            XtX += X.T @ X
            YtX += Y.T @ X
        
        # Solve ridge regression: W = Y^T X (X^T X + λI)^-1
        A = XtX + args.lambda_ * torch.eye(d_model)
        W = torch.linalg.solve(A, YtX.T).T  # (vocab_size, d_model)
        
        # Save head
        output_path = os.path.join(args.out_dir, f"{layer_key}.pt")
        torch.save(W.to(torch.bfloat16), output_path)
    
    print(f"Saved LSQ heads to {args.out_dir}")

if __name__ == "__main__":
    main()