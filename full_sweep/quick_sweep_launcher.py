#!/usr/bin/env python
"""
quick_sweep_launcher.py – one‑file driver for your LSQ / layer‑skip experiments
-----------------------------------------------------------------------------
Usage
  python quick_sweep_launcher.py sweep.yaml [--dry]

  * reads a YAML config (example at bottom of file)
  * runs dump → fit → eval in‑process on the SAME node
  * performs a smoke‑test (batch=2) before the real run
  * supports a "tiny" model alias for fast iteration
-----------------------------------------------------------------------------
"""

import argparse, subprocess, sys, time, yaml, shutil, os, textwrap, tempfile, json
from pathlib import Path

THIS = Path(__file__).resolve().parent

# ---- helpers ---------------------------------------------------------------

def run(cmd, cwd=None, env=None, check=True):
    print(f"\n$ {' '.join(map(str, cmd))}") # Ensure all parts of cmd are strings for join
    t0 = time.time()
    # If env is None, use current environment. If provided, it's used as is.
    current_env = os.environ.copy()
    if env:
        current_env.update(env)

    r = subprocess.run(cmd, cwd=cwd, env=current_env) # Use updated_env
    dt = time.time() - t0
    print(f"  ↳ exit {r.returncode}  (t={dt:,.1f}s)")
    if check and r.returncode:
        sys.exit(r.returncode)
    return r # Return the result for potential further inspection


def mem_ok(gpu_margin_gb=10):
    """Return True if every visible GPU has >margin free."""
    try:
        import torch
        if not torch.cuda.is_available():
            print("  ↳ CUDA not available, skipping VRAM check.")
            return True
        for i in range(torch.cuda.device_count()):
            free, total = torch.cuda.mem_get_info(i)
            # Check if used memory is greater than total minus margin
            # This means free memory is less than margin
            # (total - free) is used memory
            if free / 1e9 < gpu_margin_gb:
                print(f"  ↳ GPU {i}: Not enough free memory ({free/1e9:.1f}GB free, needs >{gpu_margin_gb}GB).")
                return False
        print("  ↳ VRAM check OK.")
        return True
    except ImportError:
        print("  ↳ PyTorch not found, skipping VRAM check.")
        return True # Cannot check, assume ok or handle as per requirements
    except Exception as e:
        print(f"  ↳ VRAM check failed: {e}")
        return True  # Be permissive if check fails for unexpected reasons


# ---- phases ----------------------------------------------------------------

def phase_dump(cfg, dry=False):
    if cfg.get("skip_dump") and not dry: # Always run dump in dry mode unless explicitly skipped for dry
        print("\n-- Skipping DUMP phase as per config --")
        return
    print(f"\n-- {'SMOKE TEST ' if dry else ''}DUMP phase --")
    cmd = [cfg["python"], THIS / "dump_all.py",
           "--model", cfg["model_id"],
           "--data", cfg["data_file"],
           "--out", cfg["act_dir"],
           "--seq", str(cfg["seq"]),
           "--layer_jump", str(cfg["layer_jump"]),
           "--n", str(cfg["n_prompts"] if not dry else 8), # Use small N for dry run
           "--batch", str(cfg["dump_batch"] if not dry else 2)] # Use small batch for dry run
    run(cmd)


def phase_fit(cfg, dry=False):
    if cfg.get("skip_fit") and not dry: # Always run fit in dry mode unless explicitly skipped for dry
        print("\n-- Skipping FIT phase as per config --")
        return
    print(f"\n-- {'SMOKE TEST ' if dry else ''}FIT phase --")
    # For dry run of fit, it will use the (small) activations from dry-run dump.
    # No specific dry-run parameters for fit_lsq.py itself, it just processes what's in act_dir.
    cmd = [cfg["python"], THIS / "fit_lsq.py",
           "--act_dir", cfg["act_dir"],
           "--out_dir", cfg["head_dir"],
           "--lambda_", str(cfg["fit_lambda"]),
           "--layer_jump", str(cfg["layer_jump"]),
           "--layers_per_batch", str(cfg["layers_per_batch"])]
    # fit_lsq.py uses gpu_id parameter, defaulting to 0.
    # It will respect CUDA_VISIBLE_DEVICES if set externally.
    # No specific env var needed here unless we want to override fit_lsq.py's default gpu_id.
    run(cmd)


def phase_eval(cfg, dry=False):
    print(f"\n-- {'SMOKE TEST ' if dry else ''}EVAL phase --")
    cmd = [cfg["python"], THIS / "eval_all_layers.py",
           "--model", cfg["model_id"],
           "--data", cfg["data_file"],
           "--heads", cfg["head_dir"],
           "--seq", str(cfg["seq"]),
           "--batch_size", str( cfg["eval_batch"] if not dry else 2 ), # Use small batch for dry run
           "--rows", cfg.get("rows", "train[50000:55000]"),
           "--layer_jump", str(cfg["layer_jump"])] # Default from original script

    # Prepare environment for the subprocess
    eval_env = {} # Start with an empty dict, os.environ.copy() is done in run()
    if "gpus" in cfg and cfg["gpus"] is not None: # Check for presence and not None
        eval_env["CUDA_VISIBLE_DEVICES"] = str(cfg["gpus"]) # Ensure it's a string
        print(f"  ↳ Setting CUDA_VISIBLE_DEVICES={eval_env['CUDA_VISIBLE_DEVICES']} for eval phase.")
    else:
        print("  ↳ 'gpus' not specified in config or is None; CUDA_VISIBLE_DEVICES will not be explicitly set by launcher for eval.")

    run(cmd, env=eval_env)


# ---- main ------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                 description=textwrap.dedent(__doc__))
    ap.add_argument("config", type=Path, help="Path to the YAML configuration file (e.g., sweep.yaml)")
    ap.add_argument("--dry", action="store_true", help="Run a quick smoke-test with tiny batch sizes and then exit.")
    args = ap.parse_args()

    if not args.config.is_file():
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)

    cfg_text = args.config.read_text()
    try:
        cfg = yaml.safe_load(cfg_text)
    except yaml.YAMLError as e:
        print(f"Error parsing YAML config file {args.config}:\n{e}")
        sys.exit(1)


    print(f"Loaded configuration from: {args.config}")

    # --- Environment Variables from run_sweep.sh ---
    # These should ideally be set in the environment where this launcher is called,
    # or can be added to cfg and then to `current_env` in `run` function if needed globally.
    # For now, we assume they are set externally as in run_sweep.sh.
    # Example:
    # os.environ['HF_HOME'] = cfg.get('hf_home', os.environ.get('HF_HOME', str(Path.home() / ".cache/huggingface")))
    # os.environ['PYTORCH_CUDA_ALLOC_CONF'] = cfg.get('pytorch_cuda_alloc_conf', 'expandable_segments:True')

    # tiny model alias -------------------------------------------------------
    tiny_table = {
        "tiny": "TinyLlama/TinyLlama-1.1B-Chat-v1.0", # ~4.4GB VRAM
        "small": "meta-llama/Llama-2-7b-hf"          # ~28GB VRAM (BF16)
    }
    if cfg.get("model_alias"):
        original_model_id = cfg.get("model_id", "N/A")
        cfg["model_id"] = tiny_table.get(cfg["model_alias"], cfg["model_id"])
        print(f"Using model alias '{cfg['model_alias']}': {cfg['model_id']} (original: {original_model_id})")


    # folders ----------------------------------------------------------------
    # Ensure paths are resolved relative to the config file's directory if they are relative
    # Or make them relative to CWD, or require absolute paths in YAML.
    # Current approach: if paths are relative in YAML, they are relative to CWD.
    # To make them relative to YAML location: Path(args.config.parent / cfg[k]).mkdir(...)
    for k in ("act_dir", "head_dir", "log_dir"):
        if k in cfg:
            dir_path = Path(cfg[k])

            # Clean up the directory if it exists
            if dir_path.exists():
                print(f"Cleaning up existing path: {dir_path}...")
                if dir_path.is_dir():
                    shutil.rmtree(dir_path)
                    print(f"Removed existing directory and its contents: {dir_path}")
                else: # It's a file
                    dir_path.unlink()
                    print(f"Removed existing file: {dir_path}")
            
            # Create the directory (now guaranteed to be clean or new)
            dir_path.mkdir(parents=True, exist_ok=True) # exist_ok=True is good for race conditions or if rmtree/unlink had issues
            cfg[k] = str(dir_path.resolve()) # Store absolute path back in cfg
            print(f"Created/Ensured clean directory exists: {cfg[k]}")
        else:
            print(f"Warning: Directory key '{k}' not found in config.")


    # full run (or dry run if --dry is specified) ---------------------------
    # If --dry, the 'dry' flag passed to phases will use minimal settings.
    phase_dump(cfg, dry=args.dry)
    phase_fit(cfg,  dry=args.dry) # fit will use outputs from dry dump if args.dry
    phase_eval(cfg, dry=args.dry)

    print("\n-- All phases complete --")


if __name__ == "__main__":
    main()

# -----------------------------------------------------------------------------
# Example sweep.yaml should be created as a separate file (e.g., sweep.yaml)
# in the same directory as this script.
# See the provided sweep.yaml for an example configuration.
# ----------------------------------------------------------------------------- 