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
    p.add_argument("--heads_dir", required=True, help="Directory with LSQ head files")
    p.add_argument("--n_eval", type=int, default=5000, help="Number of prompts to evaluate")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--seq_len", type=int, default=256)
    p.add_argument("--num_steps", type=int, default=16)
    args = p.parse_args()
    
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
            update_mask = {}
            for layer_idx in heads.keys():
                update_mask[layer_idx] = torch.ones(len(batch_prompts), dtype=torch.bool, device="cpu")
                actual_lengths[layer_idx] = torch.zeros(len(batch_prompts), dtype=torch.int, device="cpu")
            
            for step in range(args.num_steps):
                out = model(ids, attention_mask=attn_mask, use_cache=False, output_hidden_states=True)
                next_token = torch.argmax(out.logits[:, -1, :], dim=1, keepdim=True)
                
                # Only save data for sequences that HAVEN'T finished yet
                active_mask = ~finished  # Sequences still generating

                shard = {}
                if active_mask.any():  # If any sequences are still active
                    # Save every layer_step-th layer (only for active sequences)
                    for layer_idx in heads.keys():
                        h = out.hidden_states[layer_idx + 1]  # +1 because first is embeddings
                        h_last = h[:, -1, :].to(heads[layer_idx].device)
                        lsq_logits = h_last @ heads[layer_idx].T
                        lsq_preds = torch.argmax(lsq_logits, dim=1)
                        # Only save for sequences that haven't finished
                        shard[layer_idx] = lsq_preds.cpu()
                    
                    model_preds = torch.argmax(out.logits[:, -1, :], dim=1).cpu()
                    # Update lengths for active sequences
                    for layer_idx in heads.keys():
                        mask = (model_preds == shard[layer_idx]) & update_mask[layer_idx]
                        actual_lengths[layer_idx][mask] += 1
                        hits[layer_idx] += mask.sum()
                        total[layer_idx] += update_mask[layer_idx].sum()
                        update_mask[layer_idx] &= (model_preds == shard[layer_idx])
                        # print("=="*10, layer_idx, "=="*10)
                        # print("mask", mask)
                        # print("update_mask", update_mask[layer_idx])
                        # print("actual_lengths", actual_lengths[layer_idx])
                
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

    # Save results
    with open(os.path.join(args.heads_dir, "eval.txt"), "w") as f:
        f.write("Eval Parameters:\n")
        f.write("="*50 + "\n")
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")
    
        f.write("\n\n--- LSQ Head Accuracy ---")
        for layer_idx in sorted(heads.keys()):
            if total[layer_idx] > 0:
                acc = hits[layer_idx] / total[layer_idx]
                avg_len = seq_lens[layer_idx] / len(prompts)
                f.write(f"Layer {layer_idx:2d}: raw accuracy [{acc:.1%} ({hits[layer_idx]}/{total[layer_idx]})], avg seq len [{avg_len:.1f}]\n")
            else:
                f.write(f"Layer {layer_idx:2d}: No valid predictions\n")

if __name__ == "__main__":
    main()
