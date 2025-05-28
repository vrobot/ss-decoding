#!/usr/bin/env python
"""Extract hidden states and logits from model layers."""

import argparse
import torch
import os
from tqdm import tqdm
from utils import load_model_and_tokenizer, load_prompts

def format_prompts(prompts) -> list[str]:
    PROMPT_TEMPLATE = (
        "<|begin_of_text|><|header_start|>system<|header_end|>\n"
        "You are a helpful assistant<|eot|><|header_start|>user<|header_end|>\n"
        "{p}<|eot|>\n"
        "<|header_start|>assistant<|header_end|>\n"
    )
    return [PROMPT_TEMPLATE.format(p=p) for p in prompts]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--cache_dir", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--out_dir", default="acts")
    p.add_argument("--n_prompts", type=int, default=50000)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--seq_len", type=int, default=256)
    p.add_argument("--layer_step", type=int, default=4)
    p.add_argument("--num_steps", type=int, default=16)
    args = p.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    model, tok = load_model_and_tokenizer(args.model, cache_dir=args.cache_dir)
    prompts = format_prompts(load_prompts(args.data, args.n_prompts))
    
    print(f"Loaded {len(prompts)} prompts, processing in batches of {args.batch_size}")
    
    with torch.inference_mode():
        for i in tqdm(range(0, len(prompts), args.batch_size)):
            batch_prompts = prompts[i:i+args.batch_size]
            
            toks = tok(batch_prompts, return_tensors="pt", padding="max_length", 
                     truncation=True, max_length=args.seq_len).to(model.device)

            ids = toks['input_ids']
            attn_mask = toks['attention_mask']
            
            shard = {}
            shard["logits"] = torch.zeros(len(batch_prompts), args.num_steps, model.config.vocab_size, dtype=torch.bfloat16, device="cpu")
            for layer_idx in range(0, len(model.model.layers), args.layer_step):
                shard[f"h{layer_idx}"] = torch.zeros(len(batch_prompts), args.num_steps, model.config.hidden_size, dtype=torch.bfloat16, device="cpu")
            
            # Track which sequences have finished and their actual lengths
            finished = torch.zeros(len(batch_prompts), dtype=torch.bool, device=model.device)
            actual_lengths = torch.zeros(len(batch_prompts), dtype=torch.int, device=model.device)
            
            for step in range(args.num_steps):
                out = model(ids, attention_mask=attn_mask, use_cache=False, output_hidden_states=True)
                next_token = torch.argmax(out.logits[:, -1, :], dim=1, keepdim=True)
                
                # Only save data for sequences that HAVEN'T finished yet
                active_mask = ~finished  # Sequences still generating
                
                if active_mask.any():  # If any sequences are still active
                    # Save every layer_step-th layer (only for active sequences)
                    for layer_idx in range(0, len(model.model.layers), args.layer_step):
                        h = out.hidden_states[layer_idx + 1]  # +1 because first is embeddings
                        h_last = h[:, -1, :]
                        # Only save for sequences that haven't finished
                        shard[f"h{layer_idx}"][active_mask, step, :] = h_last[active_mask].cpu()
                    
                    # Only save logits for active sequences
                    shard["logits"][active_mask, step, :] = out.logits[active_mask, -1, :].cpu()
                    
                    # Update lengths for active sequences
                    actual_lengths[active_mask] = step + 1
                
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
            
            # Save actual lengths with the shard
            shard["lengths"] = actual_lengths.cpu()
            
            torch.save(shard, f"{args.out_dir}/batch_{i//args.batch_size:05d}.pt")

if __name__ == "__main__":
    main()
