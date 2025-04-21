# 📝 `TODO.md` — *Self‑Spec Llama‑4 Sprint*  

Put this file at the root of your repo. Work top‑to‑bottom; each checkbox is **one small, merge‑able commit** (aim ≤ 200 LOC).  
Everything assumes **conda env `llama4`**, **vLLM `nightly`** fork, **H100 80 GB**, **Llama‑4 Scout‑17B‑16E (INT4)** weights in `./model`.

---

## 📦 Milestone 0 — Baseline & Bench Harness
- [ ] **M0‑01 – clone + env**  
  ```bash
  git clone https://github.com/vllm-project/vllm.git && cd vllm && git checkout nightly
  pip install -e ".[dev]"
  ```
- [ ] **M0‑02 – pull model** (`huggingface_hub.snapshot_download`)
- [ ] **M0‑03 – add `scripts/bench.sh`** (wrap `benchmark_engine.py`, save JSON → `results/`)
- [ ] **M0‑04 – run baseline**  
  ```bash
  # micro sanity first (8p/16g) catches shape mismatches fast
  ./scripts/bench.sh baseline_sanity 8 16 --seed 0
  # full run for headline numbers
  ./scripts/bench.sh baseline 32 128 --seed 0
  # expect ~400 tok/s, record in README
  ```

---

## ☁️ Infra Bootstrap — SkyPilot + GCP Spot

> Goal: one‑line launch, auto‑retry after pre‑empt, code sync, VS Code SSH, cost guard.

| ID | task | done |
|----|------|------|
|INF‑01|`pip install -U skypilot && sky check` (laptop)|✓|
|INF‑02|Create **GCS bucket** `gs://selfspec-ckpt` (region = us‑central2)|✓|
|INF‑03|Add `infra/selfspec.yaml`<br><sup>```yaml\nresources:\n  cloud: gcp\n  region: us-central2\n  accelerators: H100:1\n  use_spot: true\n  disk_size: 80\nfile_mounts:\n  /workspace: .\nstorage_mounts:\n  /ckpt: gs://selfspec-ckpt\nsetup: |\n  conda env create -f env/environment.yml -n llama4 || true\nrun: |\n  conda activate llama4\n  python main.py --resume /ckpt/latest.pt\n```</sup>|✓|
|INF‑04|`sky launch -c llama4 infra/selfspec.yaml --env HF_TOKEN=$HF_TOKEN --env WANDB_API_KEY=$WANDB_API_KEY --idle-minutes-to-autostop 30` → VM online|✓|
|INF‑05|`sky ssh dev -- -L 10022:localhost:22` then add<br>```\nHost skydev\n  HostName 127.0.0.1\n  Port 10022\n  User sky\n``` to `~/.ssh/config`; open VS Code **Remote‑SSH: skydev**|✓|

---


That's everything missing: **Spot H100 lifecycle wrapped, automatic file sync, VSCode SSH, cost cap, and CUDA ready**. Integrate, commit, and hack away.
## ⚙️ Milestone 1 — Layer‑Skip Draft Path
- [ ] **M1‑05 – env flag `LAYERSKIP`**  
  *Patch* `vllm/engine/spec_decode.py` → expose `--layerskip` CLI flag (plumb into vLLM arg parser), if `layerskip>0` call `model.forward_partial(stop_layer=layerskip)`
- [ ] **M1‑06 – quick sweep L∈{8,12,16}** (`bench.sh skip$L`); commit CSV
  ```bash
  # cache hidden‑state tensor once; vary proj only → <30 s total
  for L in 8 12 16; do ./scripts/bench.sh skip$L 32 128 --layerskip $L --seed 0; done
  ```
- [ ] **M1‑07 – add Nsight script** (`scripts/profile.sh`)

---

## 🧩 Milestone 2 — Tiny Linear EE‑Head (LSQ Fit)
- [ ] **M2‑08 – `src/fit_lsq.py`**  
  - collect 50 k `(h12, next_id)` pairs from ShareGPT  
  - solve `H·W≈Y` via `np.linalg.lstsq` (fit LSQ in fp32 **CPU**, ~20 s)
  - *optional* 2‑min GPU SGD (1 epoch) lifts acc +4 pp
  - save `{"weight": W}` → `ee_head/ee_layer12_lsq.pt`
- [ ] **M2‑09 – wire projection**  
  *Patch* draft path: if `layerskip>0` load weight once, compute `logits = h12 @ W.T`
- [ ] **M2‑10 – bench `skip12_lsq`**; update table in README  
  *target*: ≥ 600 tok/s, ΔPPL ≤ 1.5

---

## 🚦 Milestone 3 — Self‑Spec Loop + Max‑Prob Gate
- [ ] **M3‑11 – implement longest‑prefix verify** (one helper in `spec_decode.py`)
  # gate fn chosen **once** outside hot loop → zero branch mis‑pred
- [ ] **M3‑12 – env flags**  
  - `EE_TAU` (float) → max‑prob threshold  
  - `EE_GATE=maxp`
- [ ] **M3‑13 – sweep `TAU∈{0.30,0.35,0.40}`**; log accept‑rate (accepted/total) → `results/accept.tsv`, tok/s
- [ ] **M3‑14 – update `results/summary.csv`**

---

## 📊 Milestone 4 — Var‑Entropy Gate
- [ ] **M4‑15 – add `gating.py.varentropy(logits)`**
- [ ] **M4‑16 – new env `EE_GATE=varH`, `EE_TAU_V`**
- [ ] **M4‑17 – sweep `EE_TAU_V∈{4,5,6}`**; pick τV=5 (target ≥50 % accept, ΔPPL ≤ 0.18) → aim ≥ 800 tok/s
- [ ] **M4‑18 – notebook `notebooks/roc_gate.ipynb`** (ROC of max‑prob vs varH)

---

## 🔬 Milestone 5 — Quality & Latency Validation
- [ ] **M5‑19 – add `scripts/ppl_eval.sh`** using `lm-eval-harness` on ARC, HellaSwag
  `harness --num_fewshot 0 --limit 100` # keeps run <15 min/GPU
- [ ] **M5‑20 – add `scripts/locust_load.py`** (32 users, stream 128 tokens)
  `locust -f ... --disable-log-stats` # p99 stable
- [ ] **M5‑21 – run 10 min soak, export `locust_stats.csv`**
- [ ] **M5‑22 – grafana dash screenshot**

---

## 🗂️ Milestone 6 — Refactor & PR
- [ ] **M6‑23 – move all hacks to `vllm/engine/early_exit.py`; clean imports**
  - `tests/smoke_greedy.py` (layerskip=0 ensures no‑regression)
  - `tests/smoke_skip.py` (gold 32‑tok file; CI A100‑40 GB)
- [ ] **M6‑24 – unit test** (`tests/test_early_exit.py`: ensure identical tokens when gate forced `TAU=0`)
- [ ] **M6‑25 – open PR "early‑exit self‑spec"** to upstream vLLM
- [ ] **M6‑26 – add docs snippet** (`docs/performance/early_exit.md`)

### PR Checklist
 - [ ] attach Δ‑PPL table & tokens/s bar chart in description
 - [ ] add draft release‑note snippet

---

## 📣 Milestone 7 — Blog & Broadcast
- [ ] **M7‑27 – write `blog/early_exit_llama4.md`**  
  - intro, why raw head fails  
  - 150 k‑param fix diagram  
  - embed 10‑sec GIF (htop + live token stream) top of article
  - speed table (baseline vs skip vs LSQ vs self‑spec vs varH)  
  - locust latency chart
- [ ] **M7‑28 – tweet thread + HN post** linking blog + PR

---

## 🚀 Stretch Goals (post‑merge)
- [ ] **SG‑A – autotune `LAYERSKIP` per prompt length**
- [ ] **SG‑B – bandit gate (Thompson)**
- [ ] **SG‑C – port to TensorRT‑LLM engine**
- [ ] **SG‑D – hook into router project (`llm-highperf-router`) as "local‑gpu" backend**

---

### 📈 Expected Metric Targets
| stage | tok/s | accept‑rate | ΔPPL | SM Util % |
|-------|-------|------------|------|-----------|
| baseline INT4 | ≈ 400 | – | – | ~85% |
| skip 12 + LSQ | ≈ 610 | 37 % | +1.1 | ~70% |
| self‑spec (τ 0.35) | ≈ 720 | ~40 % | +0.15 | ~65% |
| varH < 5 | **≈ 820** | ~50 % | +0.18 | ~60% |

---

### 🔑 Key Paths To Touch
```
vllm/
 └─ engine/
     ├─ spec_decode.py   
tests/
 └─ test_early_exit.py   # golden tokens
scripts/
 ├─ bench.sh
 ├─ ppl_eval.sh
 ├─ profile.sh
 └─ locust_load.py
ee_head/
 └─ ee_layer12_lsq.pt
```