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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    first_batch = torch.load(files[0], map_location=device)
    layer_keys = sorted([k for k in first_batch.keys() if k.startswith("h")])
    d_model = first_batch[layer_keys[0]].shape[2]   # Last dimension for 3D tensor
    vocab_size = first_batch["logits"].shape[2]     # Last dimension for 3D tensor
    
    print(f"Processing {len(layer_keys)} layers, d_model={d_model}, vocab_size={vocab_size}")
    
    # Process each layer
    for layer_key in tqdm(layer_keys, desc="Fitting LSQ heads"):
        XtX = torch.zeros(d_model, d_model, dtype=torch.float32, device=device)
        YtX = torch.zeros(vocab_size, d_model, dtype=torch.float32, device=device)
        
        # Accumulate over all batches
        for file_path in files:
            batch = torch.load(file_path, map_location=device)
            if layer_key not in batch:
                continue
                
            X = batch[layer_key].float()  # (B, seq_len, d_model)
            Y = batch["logits"].float()   # (B, seq_len, vocab_size)
            lengths = batch["lengths"]    # List or tensor of actual lengths
            
            # Create mask for valid positions
            batch_size, seq_len = X.shape[:2]
            mask = torch.arange(seq_len, device=device)[None, :] < lengths.to(device=device)[:, None]
            
            # Apply mask to get only valid tokens
            X_valid = X[mask]  # (total_valid_tokens, d_model)
            Y_valid = Y[mask]  # (total_valid_tokens, vocab_size)
            
            XtX += X_valid.T @ X_valid
            YtX += Y_valid.T @ X_valid
        
        # Solve ridge regression: W = Y^T X (X^T X + λI)^-1
        A = XtX + args.lambda_ * torch.eye(d_model, device=device)
        W = torch.linalg.solve(A, YtX.T).T  # (vocab_size, d_model)
        
        # Save head
        output_path = os.path.join(args.out_dir, f"{layer_key}.pt")
        torch.save(W.to(torch.bfloat16), output_path)
    
    print(f"Saved LSQ heads to {args.out_dir}")

if __name__ == "__main__":
    main()