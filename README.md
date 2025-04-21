
---

## 0 . Why does this repo exist?
Most open‑source stacks chase raw kernel speed (vLLM, TGI) but leave **routing, caching and observability** bolted on as after‑thoughts.  
Here we build a **composable control layer** that:

* **Chooses** the best model (local GPU, quantized, or API) per request  
* **Streams** tokens with sub‑20 ms TTFT  
* **Experiments** with research ideas: prefill/decode split, self‑speculative decoding, semantic cache, cost‑aware batching  
* **Exposes** rich metrics so you can _see_ every win / regression  

We start tiny → iterate → land upstream PRs.

---

## 0.5 Setup

## 0.6 Cloud Setup with SkyPilot

We use [SkyPilot](https://github.com/skypilot-org/skypilot) to manage cloud resources, specifically GCP spot instances with H100 GPUs. This gives us:

* **Cost efficiency**: Spot instances are ~70% cheaper than on-demand
* **Auto-recovery**: SkyPilot automatically resubmits jobs after preemptions
* **Easy scaling**: One command to launch/stop clusters
* **File sync**: Automatic code synchronization
* **VS Code integration**: Easy remote development

### Quick Start

Run the following command:
```bash
sky launch -c llama4 infra/selfspec.yaml --env HF_TOKEN=$HF_TOKEN --env WANDB_API_KEY=$WANDB_API_KEY --idle-minutes-to-autostop 30
```

This will 
* Provision a GCP spot instance with the specified GPU.
* Set up the necessary environment, including syncing your local code to and from GCS.
* Inject the provided Hugging Face and WandB tokens as environment variables.
* Automatically stop the instance after 30 minutes of inactivity to save costs.


## 1 . Quick 90‑second demo

```bash
# 1. clone & env
git clone https://github.com/yourname/llm-highperf-router && cd llm-highperf-router
conda env create -f env/environment.yml && conda activate llama4

# 2. pull Llama‑4 Scout‑17B‑16E (INT4)
python scripts/pull_model.py model

# 3. baseline greedy benchmark (8‑tok prompt, 16‑tok gen)
./scripts/bench.sh baseline_sanity 8 16 --seed 0

# 4. enable early‑exit self‑spec decoder
./scripts/bench.sh spec_varH5 32 128 \
  --layerskip 12 \
  --ee_weight ee_head/ee_layer12_lsq.pt \
  --ee_gate varH,5 --seed 0
```

> ✅ Expected: **≈ 820 tok/s** (2.0 × baseline) on a single H100 80 GB.  
> All core 109 B weights stay frozen; only a 150 k‑param projection is learned once.

---

## 2 . Repo layout

```
.
├── env/                  # conda + pip requirement files
├── model/                # symlink / HF snapshot (not committed)
├── ee_head/              # tiny projection checkpoint
│   └── ee_layer12_lsq.pt
├── scripts/
│   ├── pull_model.py     # HF download helper
│   ├── bench.sh          # wraps vLLM benchmark_engine.py
│   ├── fit_lsq.py        # CPU LSQ projection fit
│   ├── profile.sh        # Nsight profiler helper
│   ├── ppl_eval.sh       # lm‑eval harness wrapper
│   └── locust_load.py    # latency soak
├── vllm/ (submodule)     # fork; see engine/early_exit.py patch
├── tests/                # smoke + golden‑token tests
└── TODO.md               # granular 30‑task checklist
```

---

## 3 . High‑level architecture

```text
Client ─┬─> FastAPI Gateway (stream)
        ├─> Locust Bench (HTTP)
        ╰─> bench.sh (gRPC engine)
                 │
                 ▼
   ┌─────────────────────────┐
   │  vLLM Engine (fork)     │  – int4 weights
   │  ├─ Early‑Exit Draft    │     (layers 0‑12, LSQ head)
   │  ├─ Var‑Entropy Gate    │
   │  ├─ Verify Layers 13‑48 │
   │  ╰─ Paged KV Cache      │
   └─────────────────────────┘
                 │
                 ▼
             GPU (H100)
```

---

## 4 . Roadmap & Milestones

| ID | task | owner | status |
|----|------|-------|--------|
|M0‑03|bench script + baseline JSON|✅|
|M1‑05|`--layerskip` CLI + partial forward|🟡|
|M2‑09|LSQ projection fit (CPU) + save `.pt`|⬜|
|M3‑12|self‑spec verify loop + max‑prob gate|⬜|
|M4‑17|var‑entropy gate sweep τV=5|⬜|
|M5‑21|Locust soak (32 users, 10 min)|⬜|
|M6‑23|refactor → `early_exit.py`, unit tests|⬜|
|M6‑25|open PR to upstream vLLM|⬜|
|M7‑27|blog post + GIF demo||⬜|

*(full 30‑item checklist in `TODO.md`)*

---

## 5 . Benchmarking Cheatsheet

| command | what it does |
|---------|--------------|
|`./scripts/bench.sh baseline 32 128 --seed 0`|greedy baseline INT4|
|`./scripts/bench.sh skip12 32 128 --layerskip 12 --seed 0`|layer‑skip only|
|`./scripts/bench.sh spec_varH5 32 128 --layerskip 12 --ee_weight ee_head/ee_layer12_lsq.pt --ee_gate varH,5`|full self‑spec pipeline|

`bench.sh` writes `results/*.json` (tokens/s, accept‑rate, SM util).  
Use `python scripts/plot_results.py` to render the bar chart for your PR.

---

## 6 . Target Metrics

| metric | baseline | goal |
|--------|----------|------|
|Time‑to‑first‑token (32/128) | ~220 ms | <120 ms |
|Tokens/s | 400 | ≥ 800 |
|Δ Perplexity (ShareGPT 1 k) | — | ≤ +0.18 |
|GPU SM util | <30 % | >55 % |
|Accept‑rate | 0 % | ~50 % |

---

## 8 . Contributing

1. Fork, branch from `main`.  
2. Keep diffs ≤ 200 LOC; add/ update tests.  
3. `pre-commit run -a` must pass.  
4. Open PR; attach tokens/s chart + Δ PPL table, and provide a one‑liner changelog.

---

## 9 . References

* Apple (2024). *Recurrent Drafter: single‑model speculative decoding.*  
* NVIDIA TensorRT‑LLM docs, `forward_partial` API.  
* LayerSkip (2023). *Early‑exit techniques for transformers.*  
* vLLM PR #4630 – speculative decoding stub.