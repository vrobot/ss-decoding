#!/usr/bin/env python
"""Simple runner for the full LSQ pipeline: dump → fit → eval"""

import argparse
import subprocess
import sys
from pathlib import Path

def run_cmd(cmd):
    """Run command and exit on failure."""
    print(f"$ {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"Command failed with exit code {result.returncode}")
        sys.exit(result.returncode)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="Model ID (e.g., meta-llama/Llama-2-7b)")
    p.add_argument("--data", required=True, help="Path to ShareGPT JSON file")
    p.add_argument("--python", default="python", help="Python executable")
    
    # Data params
    p.add_argument("--n_prompts", type=int, default=50000)
    p.add_argument("--n_eval", type=int, default=5000)
    p.add_argument("--seq_len", type=int, default=256)
    p.add_argument("--layer_step", type=int, default=4)
    
    # Batch sizes
    p.add_argument("--dump_batch", type=int, default=256)
    p.add_argument("--eval_batch", type=int, default=64)
    
    # Directories
    p.add_argument("--act_dir", default="acts")
    p.add_argument("--heads_dir", default="heads")
    
    # LSQ params
    p.add_argument("--lambda_", type=float, default=1e-4)
    
    # Pipeline control
    p.add_argument("--skip_dump", action="store_true")
    p.add_argument("--skip_fit", action="store_true") 
    p.add_argument("--skip_eval", action="store_true")
    
    args = p.parse_args()
    
    # Validation
    if not Path(args.data).exists():
        print(f"Error: Data file not found: {args.data}")
        sys.exit(1)
    
    script_dir = Path(__file__).parent
    
    # 1. Dump activations
    if not args.skip_dump:
        print("\n=== DUMPING ACTIVATIONS ===")
        run_cmd([
            args.python, str(script_dir / "dump.py"),
            "--model", args.model,
            "--data", args.data,
            "--out_dir", args.act_dir,
            "--n_prompts", str(args.n_prompts),
            "--batch_size", str(args.dump_batch),
            "--seq_len", str(args.seq_len),
            "--layer_step", str(args.layer_step)
        ])
    
    # 2. Fit LSQ heads
    if not args.skip_fit:
        print("\n=== FITTING LSQ HEADS ===")
        run_cmd([
            args.python, str(script_dir / "fit.py"),
            "--act_dir", args.act_dir,
            "--out_dir", args.heads_dir,
            "--lambda_", str(args.lambda_)
        ])
    
    # 3. Evaluate heads
    if not args.skip_eval:
        print("\n=== EVALUATING HEADS ===")
        run_cmd([
            args.python, str(script_dir / "eval.py"),
            "--model", args.model,
            "--data", args.data,
            "--heads_dir", args.heads_dir,
            "--n_eval", str(args.n_eval),
            "--batch_size", str(args.eval_batch),
            "--seq_len", str(args.seq_len)
        ])
    
    print("\n✓ Pipeline complete!")

if __name__ == "__main__":
    main()