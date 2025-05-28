#!/usr/bin/env python
"""Evaluate LSQ heads against full model predictions."""

import argparse
import torch
import glob
import os
from tqdm import tqdm
from utils import load_model_and_tokenizer, load_prompts, get_last_token_indices

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--heads_dir", required=True, help="Directory with LSQ head files")
    p.add_argument("--n_eval", type=int, default=5000, help="Number of prompts to evaluate")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--seq_len", type=int, default=256)
    args = p.parse_args()
    
    model, tok = load_model_and_tokenizer(args.model)
    prompts = load_prompts(args.data, args.n_eval)[:args.n_eval]
    
    # Load all LSQ heads
    head_files = sorted(glob.glob(f"{args.heads_dir}/h*.pt"))
    if not head_files:
        raise FileNotFoundError(f"No head files found in {args.heads_dir}")
    
    heads = {}
    for head_file in head_files:
        layer_name = os.path.basename(head_file).replace('.pt', '')
        layer_idx = int(layer_name[1:])  # Extract number from "h12"
        heads[layer_idx] = torch.load(head_file, map_location="cpu")
    
    print(f"Loaded {len(heads)} LSQ heads: {sorted(heads.keys())}")
    print(f"Evaluating on {len(prompts)} prompts")
    
    # Track accuracy for each layer
    hits = {layer: 0 for layer in heads.keys()}
    total = {layer: 0 for layer in heads.keys()}
    
    with torch.no_grad():
        for i in tqdm(range(0, len(prompts), args.batch_size)):
            batch_prompts = prompts[i:i+args.batch_size]
            
            ids = tok(batch_prompts, return_tensors="pt", padding="max_length",
                     truncation=True, max_length=args.seq_len).to(model.device)
            
            last_idx = get_last_token_indices(ids["attention_mask"])
            
            # Get model outputs
            out = model(**ids, use_cache=False, output_hidden_states=True)
            
            # Gold labels from full model
            gold_logits = out.logits[torch.arange(out.logits.size(0)), last_idx]
            gold_labels = gold_logits.argmax(dim=-1)
            
            # Evaluate each LSQ head
            for layer_idx, head_weight in heads.items():
                if layer_idx + 1 >= len(out.hidden_states):
                    continue
                    
                # Get hidden state at this layer
                h = out.hidden_states[layer_idx + 1]  # +1 for embeddings
                h_last = h[torch.arange(h.size(0)), last_idx]
                
                # Predict with LSQ head
                head_weight = head_weight.to(h_last.device, dtype=h_last.dtype)
                lsq_logits = h_last @ head_weight.T
                lsq_labels = lsq_logits.argmax(dim=-1)
                
                # Count correct predictions (excluding EOS)
                valid_mask = gold_labels != tok.eos_token_id
                if valid_mask.any():
                    correct = (gold_labels[valid_mask] == lsq_labels[valid_mask]).sum().item()
                    hits[layer_idx] += correct
                    total[layer_idx] += valid_mask.sum().item()
    
    # Print results
    print("\n--- LSQ Head Accuracy ---")
    for layer_idx in sorted(heads.keys()):
        if total[layer_idx] > 0:
            acc = hits[layer_idx] / total[layer_idx]
            print(f"Layer {layer_idx:2d}: {acc:.1%} ({hits[layer_idx]}/{total[layer_idx]})")
        else:
            print(f"Layer {layer_idx:2d}: No valid predictions")

if __name__ == "__main__":
    main()