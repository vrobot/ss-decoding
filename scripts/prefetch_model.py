import os, sys, pathlib
from huggingface_hub import snapshot_download

model_id   = sys.argv[1]
token      = os.getenv("HF_TOKEN")     # set via --env
cache_dir  = os.getenv("HF_HOME")

# ---------- fast‑exit if already cached ----------
try:
    existing = snapshot_download(
        model_id,
        cache_dir=cache_dir,
        local_files_only=True,         # don't hit the wire
        token=token,
    )
    print(f"✅  {model_id} already cached → {existing}")
    sys.exit(0)
except FileNotFoundError:             # nothing cached yet
    pass
# any other exception (e.g. permission) will fall through
# -----------------------------------------------

dst = pathlib.Path(
    snapshot_download(
        model_id,
        cache_dir=cache_dir,
        token=token,
        resume_download=True           # resume if spot‑instance died mid‑pull
    )
)
print(f"✅  Prefetched {model_id} → {dst}")
