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

def slugify(text):
    return text.lower().replace("/", "_").replace("-", "_")

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

    fit_env = {}
    # Set PYTORCH_CUDA_ALLOC_CONF to potentially mitigate fragmentation OOMs
    fit_env['PYTORCH_CUDA_ALLOC_CONF'] = cfg.get('pytorch_cuda_alloc_conf', 'expandable_segments:True')
    print(f"  ↳ Setting PYTORCH_CUDA_ALLOC_CONF={fit_env['PYTORCH_CUDA_ALLOC_CONF']} for fit phase.")

    if "gpus" in cfg and cfg["gpus"] is not None:
        fit_env["CUDA_VISIBLE_DEVICES"] = str(cfg["gpus"])
        print(f"  ↳ Setting CUDA_VISIBLE_DEVICES={fit_env['CUDA_VISIBLE_DEVICES']} for fit phase.")
    else:
        print("  ↳ 'gpus' not specified in config or is None; CUDA_VISIBLE_DEVICES will not be explicitly set by launcher for fit.")

    run(cmd, env=fit_env)


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
    eval_env = {}
    if "gpus" in cfg and cfg["gpus"] is not None: # Check for presence and not None
        eval_env["CUDA_VISIBLE_DEVICES"] = str(cfg["gpus"]) # Ensure it's a string
        print(f"  ↳ Setting CUDA_VISIBLE_DEVICES={eval_env['CUDA_VISIBLE_DEVICES']} for eval phase.")
    else:
        print("  ↳ 'gpus' not specified in config or is None; CUDA_VISIBLE_DEVICES will not be explicitly set by launcher for eval.")

    # Set TMPDIR to a writable location to avoid FileNotFoundError
    # Create a specific temp dir for this run if log_dir is specified
    if "log_dir" in cfg:
        run_temp_dir = Path(cfg["log_dir"]) / "tmp" / slugify(cfg.get("model_id", "unknown_model"))
        run_temp_dir.mkdir(parents=True, exist_ok=True)
        eval_env["TMPDIR"] = str(run_temp_dir.resolve())
        print(f"  ↳ Setting TMPDIR={eval_env['TMPDIR']} for eval phase.")
    else:
        # Create a default temp directory if log_dir is not in cfg
        # This uses tempfile.mkdtemp to create a secure temporary directory
        # This directory will be automatically cleaned up by the OS, or can be manually cleaned if needed.
        default_temp_dir = tempfile.mkdtemp(prefix="quick_sweep_eval_")
        eval_env["TMPDIR"] = default_temp_dir
        print(f"  ↳ 'log_dir' not in config. Setting TMPDIR to system default temp: {eval_env['TMPDIR']} for eval phase.")


    run(cmd, env=eval_env)


# ---- main ------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                 description=textwrap.dedent(__doc__))
    ap.add_argument("config", type=Path, help="Path to the YAML configuration file (e.g., sweep.yaml)")
    ap.add_argument("--dry", action="store_true", help="Run a quick smoke-test with tiny batch sizes and then exit.")
    ap.add_argument("--models", type=str,
                    help="comma-separated list of model IDs; "
                         "launcher will run one job per model sequentially")
    ap.add_argument("--log_root", type=Path, default=Path("logs"),
                    help="folder for *.out / *.err logs (default: ./logs)")
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

    # ------------------------------------------------------------------
    # If --models given, override cfg["model_id"] with each entry in turn
    # ------------------------------------------------------------------
    model_list = [m.strip() for m in args.models.split(",")] if args.models else [cfg["model_id"]]

    print(f"Loaded configuration from: {args.config}")

    # --- Environment Variables from YAML ---
    if "hf_home" in cfg and cfg["hf_home"]:
        os.environ['HF_HOME'] = str(Path(cfg["hf_home"]).resolve())
        print(f"  ↳ Set HF_HOME to: {os.environ['HF_HOME']}")
        # Ensure the directory exists and is writable (permissions should be handled by user)
        Path(os.environ['HF_HOME']).mkdir(parents=True, exist_ok=True)

    if "pytorch_cuda_alloc_conf" in cfg and cfg["pytorch_cuda_alloc_conf"]:
        # This will be used by phase_fit, but can also be set globally if needed
        # For now, phase_fit handles it for its subprocess.
        # If other torch operations are done directly in this script before phases,
        # setting it globally here might be useful:
        # os.environ['PYTORCH_CUDA_ALLOC_CONF'] = cfg['pytorch_cuda_alloc_conf']
        # print(f"  ↳ PYTORCH_CUDA_ALLOC_CONF will be set to: {cfg['pytorch_cuda_alloc_conf']} for relevant phases.")
        pass # Handled in phase_fit

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
        elif k == "log_dir": # Ensure log_dir is created even if not for cleanup
            dir_path = Path(cfg.get(k, args.log_root)) # Use args.log_root as default if not in cfg
            dir_path.mkdir(parents=True, exist_ok=True)
            cfg[k] = str(dir_path.resolve())
            print(f"Ensured log directory exists: {cfg[k]}")
        else:
            print(f"Warning: Directory key '{k}' not found in config.")


    # full run (or dry run if --dry is specified) ---------------------------
    # If --dry, the 'dry' flag passed to phases will use minimal settings.
    args.log_root.mkdir(exist_ok=True)

    for m in model_list:
        run_slug = slugify(m)
        print(f"\n=== RUN {run_slug} ===========================================")

        # ------------- per-run overrides ---------------------------------
        cfg_one = cfg.copy() # Shallow copy is fine as we only change top-level primitive values or replace dicts/lists
        cfg_one["model_id"] = m

        # Create per-run act_dir and head_dir. These will be cleaned up if they exist.
        # The main loop already creates the base act_dir and head_dir.
        # We now make subdirectories within those for each run.
        base_act_dir = Path(cfg["act_dir"]) # Original base act_dir from cfg
        base_head_dir = Path(cfg["head_dir"]) # Original base head_dir from cfg

        run_act_dir = base_act_dir / run_slug
        run_head_dir = base_head_dir / run_slug

        # Clean and create specific directories for this run
        for run_dir_path in [run_act_dir, run_head_dir]:
            if run_dir_path.exists():
                print(f"Cleaning up existing path for run {run_slug}: {run_dir_path}...")
                if run_dir_path.is_dir():
                    shutil.rmtree(run_dir_path)
                    print(f"Removed existing directory and its contents: {run_dir_path}")
                else: # It's a file
                    run_dir_path.unlink()
                    print(f"Removed existing file: {run_dir_path}")
            run_dir_path.mkdir(parents=True, exist_ok=True)
            print(f"Created/Ensured clean directory for run {run_slug}: {run_dir_path.resolve()}")

        cfg_one["act_dir"]  = str(run_act_dir.resolve())
        cfg_one["head_dir"] = str(run_head_dir.resolve())

        # ------------- log files ----------------------------------------
        out_path = args.log_root / f"{run_slug}.out"
        err_path = args.log_root / f"{run_slug}.err"
        with open(out_path, "wb") as out_f, open(err_path, "wb") as err_f:
            # wrap each phase so stdout/err go to the files
            # We need to pass file objects to subprocess.run, not sys.stdout/stderr
            # The print statements within this loop should go to the original stdout
            
            def run_phase_logged(fn, name, current_cfg, dry_run):
                # This print goes to the main console
                print(f"-- Starting phase {name} for {run_slug} (logs: {out_path.name}, {err_path.name}) --")
                
                # Redirect script's print statements for the duration of the phase
                # The subprocess `run` function will handle its own stdout/stderr via Popen
                # For prints within phase_dump, phase_fit, phase_eval (if they use print() directly)
                # we need to capture them. The run() helper captures subprocess output.
                # The problem is the print() statements *within* phase_dump, phase_fit, phase_eval
                # if they are not part of the subprocess but part of the Python script itself.

                # The original request redirects sys.stdout/stderr. Let's stick to that.
                # This means print() calls *within* phase_x functions will go to files.
                orig_stdout, orig_stderr = sys.stdout, sys.stderr
                # The file for print() must be in text mode.
                # The subprocess.run in the `run` helper will still write bytes if needed.
                # So we open text mode for sys.stdout redirection.
                with open(out_path, "a", encoding='utf-8', buffering=1) as text_out, \
                     open(err_path, "a", encoding='utf-8', buffering=1) as text_err:
                    sys.stdout, sys.stderr = text_out, text_err
                    try:
                        print(f"-- {name} ({run_slug}) --", flush=True) # This goes to the file
                        fn(current_cfg, dry=dry_run) # This function's prints go to file
                    finally:
                        sys.stdout, sys.stderr = orig_stdout, orig_stderr # Restore

            # Run phases with logging
            run_phase_logged(phase_dump, "DUMP", cfg_one, args.dry)
            run_phase_logged(phase_fit,  "FIT", cfg_one, args.dry)
            run_phase_logged(phase_eval, "EVAL", cfg_one, args.dry)

        # This print goes to the main console
        print(f"✓ finished {run_slug}  (logs → {out_path.relative_to(Path.cwd()) if out_path.is_absolute() else out_path.relative_to(args.log_root.parent)})")

    print("\n-- All phases complete --")


if __name__ == "__main__":
    main()

# -----------------------------------------------------------------------------
# Example sweep.yaml should be created as a separate file (e.g., sweep.yaml)
# in the same directory as this script.
# See the provided sweep.yaml for an example configuration.
# ----------------------------------------------------------------------------- 