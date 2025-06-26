#!/usr/bin/env python
"""Evaluate LSQ heads against full model predictions."""

import argparse
import torch
import glob
import os
from tqdm import tqdm
from utils import load_model_and_tokenizer, load_prompts, format_prompts

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--cache_dir", default=None)
    p.add_argument("--data", required=True)
    p.add_argument("--out_dir", default="evals")
    p.add_argument("--heads_dir", required=True, help="Directory with LSQ head files")
    p.add_argument("--n_eval", type=int, default=5000, help="Number of prompts to evaluate")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--seq_len", type=int, default=256)
    p.add_argument("--num_steps", type=int, default=16)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    
    model, tok = load_model_and_tokenizer(args.model, cache_dir=args.cache_dir)
    prompts = load_prompts(args.data, args.n_eval, split="test")
    ### UNCOMMENT THESE LINES WHEN RUNNING INSTRUCT MODELS
    # prompts = format_prompts(prompts)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load all LSQ heads
    head_files = sorted(glob.glob(f"{args.heads_dir}/h*.pt"))
    if not head_files:
        raise FileNotFoundError(f"No head files found in {args.heads_dir}")
    
    heads = {}
    for head_file in head_files:
        layer_name = os.path.basename(head_file).replace('.pt', '')
        layer_idx = int(layer_name[1:])  # Extract number from "h12"
        heads[layer_idx] = torch.load(head_file, map_location=device)
    
    print(f"Loaded {len(heads)} LSQ heads: {sorted(heads.keys())}")
    print(f"Evaluating on {len(prompts)} prompts")
    
    # Track accuracy for each layer
    hits = {layer: 0 for layer in heads.keys()}
    total = {layer: 0 for layer in heads.keys()}
    seq_lens = {layer: 0 for layer in heads.keys()}
    
    with torch.inference_mode():
        for i in tqdm(range(0, len(prompts), args.batch_size)):
            batch_prompts = prompts[i:i+args.batch_size]
            
            toks = tok(batch_prompts, return_tensors="pt", padding="max_length", 
                     truncation=True, max_length=args.seq_len).to(model.device)

            ids = toks['input_ids']
            attn_mask = toks['attention_mask']

            # Track which sequences have finished and their actual lengths
            finished = torch.zeros(len(batch_prompts), dtype=torch.bool, device=model.device)
            actual_lengths = {}
            shard = {}

            shard["next_token"] = torch.zeros(len(batch_prompts), args.num_steps, dtype=torch.int64, device="cpu")

            for layer_idx in heads.keys():
                actual_lengths[layer_idx] = torch.zeros(len(batch_prompts), dtype=torch.int, device="cpu")
                shard[f"n{layer_idx}"] = torch.zeros(len(batch_prompts), args.num_steps, dtype=torch.int64, device="cpu")
                shard[f"e{layer_idx}"] = torch.zeros(len(batch_prompts), args.num_steps, dtype=torch.float32, device="cpu")

            
            for step in range(args.num_steps):
                out = model(ids, attention_mask=attn_mask, use_cache=False, output_hidden_states=True)
                next_token = torch.argmax(out.logits[:, -1, :], dim=1, keepdim=True)
                
                # Only save data for sequences that HAVEN'T finished yet
                active_mask = ~finished  # Sequences still generating

                if active_mask.any():  # If any sequences are still active
                    # Save every layer_step-th layer (only for active sequences)
                    for layer_idx in heads.keys():
                        h = out.hidden_states[layer_idx + 1]  # +1 because first is embeddings
                        h_last = h[active_mask, -1, :].to(heads[layer_idx].device)
                        lsq_logits = h_last @ heads[layer_idx].T
                        lsq_probs = lsq_logits.float().softmax(dim=-1)
                        lsq_entropy = (lsq_probs*torch.log(lsq_probs.clamp(min=1e-12))).sum(dim=-1)
                        shard[f"e{layer_idx}"][active_mask, step] = lsq_entropy.cpu()
                        lsq_preds = torch.argmax(lsq_logits, dim=1)
                        shard[f"n{layer_idx}"][active_mask, step] = lsq_preds.cpu()
                    
                    shard["next_token"][active_mask, step] = next_token.squeeze()[active_mask].cpu()
                
                # Mark finished sequences (simplified!)
                finished |= (next_token.squeeze() == tok.eos_token_id)
                
                # If all finished, stop early
                if finished.all():
                    print(f"Early stop at step {step+1} - all sequences finished")
                    break
                
                # Replace EOS tokens with pad tokens for finished sequences
                next_token[finished] = tok.pad_token_id
                
                # Continue generation
                ids = torch.cat((ids, next_token), dim=1)
                attn_mask = torch.cat((attn_mask, torch.ones(attn_mask.shape[0], 1, device=attn_mask.device)), dim=1)

            for layer_idx in heads.keys():
                seq_lens[layer_idx] += actual_lengths[layer_idx].sum()
            # breakpoint()

            torch.save(shard, f"{args.out_dir}/batch_{i//args.batch_size:05d}.pt")

if __name__ == "__main__":
    main()
