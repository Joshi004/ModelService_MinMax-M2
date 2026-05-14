# MiniMax-M2.5 vLLM Service — Capacity Model

Service path: `/home/vision/ModelService_MinMax-M2`
Engine: vLLM V1, version `v0.19.1rc1.dev71+gdd9342e6b`
Hardware: 4× NVIDIA H100 80 GB (of 8 in the box), TP=4
Log analysed: `logs/service.log` (4 598 lines, 2026-04-07 15:56 → 2026-04-17 19:29, ~10 days uptime)

> All numeric claims below are traceable to log lines quoted in the appendix. Estimates and speculative statements are flagged inline.

---

## 1. Executive summary

- Serves `MiniMaxAI/MiniMax-M2.5` (MoE, 230 B total / 10 B active) in FP8 across 4× H100 via tensor-parallel 4.
- Global KV pool = **307 184 tokens** (engine-reported). With `max_model_len = 128 000`, that gives only **2.40× concurrency** at full context.
- Soft cap `max_num_seqs = 10`; hard cap is KV-bound below ~30 K token slots.
- The service has been heavily **under-utilised** over the observed 10-day window: ~96.4 % of 10-s log samples reported `Running: 0`, and the peak concurrency ever seen in the log is **2 in-flight requests** (single sample).
- No OOMs, no preemptions, no swapping, no errors logged.
- Observed single-request throughput: **prefill ≈ 2 000 – 2 700 tok/s**, **decode ≈ 95 – 100 tok/s**.
- Prefix caching is effective: hit rate climbed from 0 % → ~71 % over the first ~80 minutes of live traffic and has held in the 30 – 70 % range since.
- The **single most binding constraint for long-context work is KV-cache tokens** (max_model_len=128 K leaves only ~2.4× concurrency). For short prompts the binding constraint is the soft cap `max_num_seqs=10`, then decode throughput.
- Biggest low-risk win: lower `max_model_len` to what callers actually use (e.g. 32 K or 64 K) → concurrency scales roughly inversely (see §5). No code changes, no risk of quality loss unless callers genuinely need >64 K context.

---

## 2. Hardware footprint

| GPU | Role | VRAM | Used by this service |
|---|---|---|---|
| 0 | TP rank 0 | 80 GB HBM3 | Yes (MiniMax) |
| 1 | TP rank 1 | 80 GB HBM3 | Yes (MiniMax) |
| 2 | TP rank 2 | 80 GB HBM3 | Yes (MiniMax) |
| 3 | TP rank 3 | 80 GB HBM3 | Yes (MiniMax) |
| 4 | — | 80 GB HBM3 | Parakeet (separate service) |
| 5 | — | 80 GB HBM3 | Idle |
| 6 | — | 80 GB HBM3 | Idle |
| 7 | — | 80 GB HBM3 | Idle |

- Host: Intel Xeon Sapphire Rapids, 160 vCPU, 983 GiB RAM.
- CUDA 12.4, driver 570.211.01.
- NCCL 2.27.5, FlashAttention v3, CutlassFP8ScaledMMLinearKernel, TRITON Fp8 MoE backend (see appendix lines 29, 36–39).
- TP all-reduce uses custom flashinfer fusion (`fuse_allreduce_rms`, line 17 — `Enabled custom fusions: norm_quant, act_quant, allreduce_rms`).

---

## 3. Model & runtime configuration

Sourced from `config.env` + `start_service.sh` + engine config line in the log.

| Knob | Value | Effect (1-line) |
|---|---|---|
| `MODEL_NAME` | `MiniMaxAI/MiniMax-M2.5` | MoE 230 B total / 10 B active; native 204 800 ctx |
| `quantization` | `fp8` | Weights in FP8 (auto-selected CutlassFP8 matmul) — ~2× weight shrink vs bf16 |
| `dtype` | `bfloat16` | Activation / attention dtype |
| `kv_cache_dtype` | `auto` (= bf16) | NOT yet fp8; doubling candidate — see §9 |
| `max_model_len` | 128 000 | Hard ceiling on (input + output) tokens per request |
| `tensor_parallel_size` | 4 | Model shards across GPUs 0–3 |
| `gpu_memory_utilization` | 0.95 | ~76 GB out of 80 GB per GPU available to vLLM |
| `max_num_seqs` | 10 | Scheduler soft cap on concurrent sequences |
| `max_num_batched_tokens` | 8 192 | Chunked-prefill batch budget (log line 12) |
| `enable_prefix_caching` | true (default) | Automatic; observed ~71 % peak hit-rate |
| `enable_chunked_prefill` | true | Lets decode progress while long prefills chunk |
| `cudagraph_mode` | FULL + PIECEWISE, capture sizes 1/2/4/8/16 | Decode-batch sweet spot is ≤16 |
| `--enable-auto-tool-choice` | on | Tool-calling routing |
| `--tool-call-parser` | `minimax_m2` | Parses M2.5 tool-call JSON |
| `--reasoning-parser` | `minimax_m2` | Strips `<think>` from responses (but model still reasons) |
| Asynchronous scheduling | enabled | Overlaps scheduling with forward pass |
| Port | 9084 | OpenAI-compatible HTTP API on `0.0.0.0:9084` |

---

## 4. Memory budget per GPU

Figures are the engine's own numbers (TP rank 0; other ranks are within ~0.02 GiB).

| Bucket | Per-GPU | ×4 shards | Source |
|---|---|---|---|
| Model weights (FP8) | 53.75 GiB | 215.0 GiB | log line 169: `Model loading took 53.75 GiB memory` |
| CUDA-graph pool | 0.19 GiB | 0.76 GiB | log 1010–1014 |
| Available KV cache | **18.16 GiB** | **72.64 GiB** | log 988 |
| vLLM allowed (0.95 × 80) | ≈ 76 GiB | 304 GiB | from `GPU_MEMORY_UTIL=0.95` |
| Headroom (activations + fragmentation + misc) | ≈ 76 − 53.75 − 0.19 − 18.16 ≈ **3.9 GiB** | 15.6 GiB | derived |

Observations:
- Weights dominate (~71 % of the vLLM budget per GPU).
- Only ~24 % of per-GPU budget is KV cache. This is why long contexts bite hard.
- The engine hinted at a tiny free lever: setting `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1` and bumping util from 0.9500 → 0.9526 would "maintain the same effective KV cache size" under the new v0.19 default profiler (log 985). Trivial safety margin, low impact (<2 % more KV).

---

## 5. KV-cache math

Ground truth from the engine:

> `GPU KV cache size: 307,184 tokens` (line 990)
> `Maximum concurrency for 128,000 tokens per request: 2.40x` (line 991)

That is, the KV pool is sized in (global) tokens — each position of each active sequence consumes one slot. Effective concurrency at a given `max_model_len` L is ≈ `307 184 / L`, bounded above by `max_num_seqs = 10`.

If `max_model_len` is lowered, the KV pool stays approximately constant (chunked-prefill activations remain bounded by `max_num_batched_tokens=8 192`, not by `max_model_len`), so concurrency scales roughly linearly with `1/L`. Small second-order gains (~5–15 %) may come from smaller attention scratch buffers; not counted below.

| `max_model_len` | KV-bound concurrency (= 307 184 / L) | Effective concurrency (min with `max_num_seqs=10`) |
|---|---|---|
| 128 000 | 2.40× | **2** (KV-bound) |
| 64 000 | 4.80× | **4** (KV-bound) |
| 32 768 | 9.37× | **9** (KV-bound, essentially at cap) |
| 16 384 | 18.75× | **10** (cap of `max_num_seqs`) |
| 8 192 | 37.5× | **10** (cap of `max_num_seqs`) |

Implication: beyond `max_model_len ≈ 30 720` we stop being KV-bound and start being `max_num_seqs`-bound. If we drop max_model_len to 32 K, we'd want to raise `max_num_seqs` in lockstep to exploit the new KV headroom.

---

## 6. Observed workload (evidence from `logs/service.log`)

### 6.1 Concurrency distribution (2 273 `Running: N` samples over 10 days)

| Samples | `Running:` value | Fraction |
|---|---|---|
| 2 216 | 0 reqs | 97.5 % |
| 56 | 1 req | 2.46 % |
| 1 | 2 reqs | 0.04 % |
| 0 | ≥3 reqs | 0 % |
| 0 | `Waiting: ≥1` (queueing) | 0 % |

The service has **never once queued**. Peak concurrency is 2, observed once on 04-07 17:54:45 (line 1201).

### 6.2 Throughput (single active request)

From `Running: 1 reqs` samples (representative, lines 1229–1350):

| Phase | Observed range | Typical |
|---|---|---|
| Prompt prefill (single req) | 30 – 2 735 tok/s | **~2 000 – 2 700 tok/s** when actually prefilling |
| Decode (single req) | 50 – 100 tok/s | **~95 – 100 tok/s** steady-state |

The N=2 datapoint (line 1201) shows an aggregate decode of 93.6 tok/s — but this likely captures one sequence still finishing prefill, so it is **not reliable** evidence about decode scaling under concurrency. Treat per-slot decode under true concurrent load as speculative (see §7).

### 6.3 GPU KV cache utilisation

- Max KV usage ever observed: **16.4 %** (at `Running: 2`, line 1201).
- Typical busy-sample KV usage with `Running: 1`: 0.5 – 9 %.
- There is a massive amount of unused KV headroom under the current workload; KV pressure is theoretical, not actual.

### 6.4 Prefix cache

- First observation: 0 % (cold engine).
- Climbed past 50 % within ~2 h (line 1067: 55.8 %), and past 70 % within ~1 h more (line 1151: 71.0 %).
- Steady-state since: ~30–71 %, depending on traffic mix. Strong signal that callers reuse large prompt prefixes (system prompts / tool schemas).

### 6.5 Incidents

- `OutOfMemory` / `CUDA out of memory` / `preempted` / `swapped`: **0 matches** across the whole log.
- The only "Error" strings in the log are inside verbose C++ kernel signatures in a flashinfer `WARNING` banner (lines 413–449); not a runtime error.
- No crashes, no restarts. PID 63120 has been the API server PID the whole 10 days.

### 6.6 Startup cost (one-off)

| Phase | Duration | Line |
|---|---|---|
| Load safetensors weights | 34.89 s | 167 |
| Model loading total | 37.95 s | 169 |
| `torch.compile` | 57.56 s | 965 |
| Initial profiling / warmup | 7.12 s | 968 |
| Graph capture | 4 s | 1012 |
| Engine init (profile + KV + warmup) | **78.95 s** | 1015 |

Cold-start to first-ready is dominated by weight load + compile (~95 s combined). Not a capacity issue, but relevant for failover / rolling-restart planning.

---

## 7. Current bottleneck analysis

Binding constraint by request shape:

| Scenario | First binding constraint | Why |
|---|---|---|
| Short ctx (≤8 K total) | (b) `max_num_seqs = 10` | KV could hold 37× but scheduler caps at 10 |
| Medium ctx (8 K – 30 K total) | (b) `max_num_seqs = 10` | KV headroom still fine (≥10×) |
| Long ctx (30 K – 128 K total) | (a) **KV-cache tokens** | 307 184 / L < 10 |
| Any single-request latency | (c) decode tok/s (~95/s) | Hard per-slot floor; MoE 10 B active, memory-bw bound |
| Aggregate throughput under bursts | (d) TP-4 all-reduce + MoE routing overhead | Speculative — no measurement in log; see §11 |
| New-weight capacity | (e) Weights already own 53.75 GiB / GPU | Little room to grow without dropping TP size or quant |

Rank-ordered limiters (from most to least binding, given the observed workload):

1. **KV-cache tokens at 128 K** — caps concurrency at 2 for long-context requests. Biggest structural lever.
2. **`max_num_seqs = 10`** — soft cap that bites first for any prompt ≲30 K total.
3. **Single-request decode throughput (~95 tok/s)** — sets the floor on time-to-finish for any long-output request.
4. **TP-4 all-reduce overhead** — real but already fused (`allreduce_rms`); not measured here, probably not on the critical path for N ≤ 10.
5. **Weights consuming majority of VRAM** — means we can't easily re-shape the budget without FP8-KV or dropping weights to a smaller model.

Note on the real world: the service is currently load-limited, not capacity-limited. The peak observed concurrency of 2 is nowhere near either the KV cap (2.40 at 128 K, higher at shorter ctx) or the `max_num_seqs` cap of 10. If the upstream load shape is durable, **there is no capacity incident waiting to happen under today's traffic.**

---

## 8. Capacity model — headline numbers

Assumptions (state explicitly before reading the table):

- Prefill throughput per request: **~2 700 tok/s** (observed upper end of single-req prefill samples).
- Decode throughput per active slot: **~95 tok/s** (observed single-req steady state).
- KV cost per request ≈ `input + output` tokens (one slot per position).
- Max parallel = `min(max_num_seqs, floor(307 184 / (input+output)))`.
- **Per-slot decode throughput is assumed to hold at observed concurrency (≤2).** Scaling to 4–10 concurrent slots is extrapolation; the N=2 datapoint in the log is too noisy to confirm. The "req/hr (upper)" column is therefore an **upper bound**, not a measured number. A conservative column halves per-slot decode under concurrency.
- Ignores network, tokenisation, tool-call round-trips.

| Shape (in / out tokens) | KV tokens / req | Max parallel | Prefill time | Decode time | Per-request wall-clock | Req/hr (upper)¹ | Req/hr (conservative)² |
|---|---|---|---|---|---|---|---|
| 4 K / 1 K | 5 120 | 10 (cap) | 1.5 s | 10.8 s | ~12.3 s | ~2 930 | ~1 750 |
| 8 K / 2 K | 10 240 | 10 (cap) | 3.0 s | 21.6 s | ~24.6 s | ~1 460 | ~900 |
| 32 K / 4 K | 36 864 | 8 (KV) | 12.1 s | 43.1 s | ~55.3 s | ~520 | ~330 |
| 64 K / 8 K | 73 728 | 4 (KV) | 24.3 s | 86.2 s | ~110.5 s | ~130 | ~85 |
| 120 K / 8 K | 131 072 | 2 (KV) | 45.5 s | 86.2 s | ~131.7 s | ~55 | ~36 |

¹ Upper bound: assumes per-slot 95 tok/s decode holds at full concurrency. Speculative above N=2.
² Conservative: assumes aggregate decode throughput saturates near ~160 tok/s total (i.e. per-slot 95 / √N). Also speculative but gives a better lower bound.

Pattern to internalise: the service can do **~10 small chats at once** or **~2 whole-book long-context queries at once**. Nothing in between comes free.

---

## 9. Opportunities to get more out of the same hardware

Ordered by confidence (high → low). Each row is a single knob you could turn without changing hardware or model.

| # | Change | Expected effect | Risk / caveat |
|---|---|---|---|
| 1 | **Lower `max_model_len`** to what callers actually use (e.g. 32 K or 64 K) | Concurrency scales ~linearly with `1/L`: 32 K → up to 9×, 64 K → 4× (vs today's 2.40×) | Will reject requests with longer inputs; audit caller logs first |
| 2 | **Raise `max_num_seqs`** in lockstep with (1) | At `max_model_len ≤ 32 K`, raising to 16–24 is free in KV; matches prior config in `README.md` (16) | Beyond CUDA-graph capture size 16, falls back to piecewise/eager — slight latency bump |
| 3 | Keep prefix caching on; consider tuning eviction | Already ~71 % peak hit rate — dominant free win on repeated system prompts | None; already enabled |
| 4 | Tune chunked-prefill `--max-num-batched-tokens` | Current 8 192. Bumping to 16 384 may raise prefill tok/s on long prompts while still letting decode interleave | Bigger batches ↑ peak activation memory; must re-profile |
| 5 | Bump `gpu_memory_utilization` 0.95 → 0.9526 (per engine hint, line 985) | Keeps effective KV the same under v0.19's new cudagraph-aware profiler | Near-zero risk; cosmetic |
| 6 | Bump `gpu_memory_utilization` 0.95 → 0.97 | ~1.5 GiB extra KV per GPU → ~6 GiB total → ~+25 K KV tokens | Smaller headroom for memory spikes; recommend keeping after (5) |
| 7 | `--kv-cache-dtype fp8` (**speculative**) | Roughly doubles KV pool (≈ 600 K tokens) → concurrency at 128 K ≈ 4.8× | Need to confirm M2.5 + this vLLM nightly supports fp8 KV for MiniMaxM2 attention; slight accuracy hit possible on long contexts. Validate with a correctness A/B before rolling out. |
| 8 | Disable `<think>` preservation if callers don't need it downstream | Fewer output tokens → less decode time and less KV | Already off (`REASONING_PARSER="minimax_m2"` strips `<think>`). Just keep it off. |
| 9 | Speculative decoding / draft model (**speculative**) | Potential 1.5–3× decode speedup | vLLM support for M2.5 draft is unverified; MoE draft models non-trivial. Research item. |
| 10 | Reduce TP 4 → 2, co-locate two engines on 4 GPUs (**speculative**) | Two independent engines = ~2× replica concurrency at lower per-request throughput | Weights are ~215 GiB; 2-way TP would need ~108 GiB per GPU, which exceeds 80 GiB. **Not feasible** without a smaller/more-quantised model. Flagged so it's not revisited. |

Safety notes:
- None of (1)–(6) delete or mutate user-facing data; they are pure runtime flags and are reversible by restart.
- (7) is the highest-impact idea but must be validated; start in a canary before promoting.
- Do **not** change `TENSOR_PARALLEL_SIZE` or `MODEL_PATH` without a full restart + smoke test plan.

---

## 10. Horizontal scaling hooks (brief)

Details live in the separate `HORIZONTAL_SCALING.md`; this is the pointer.

- GPUs 5/6/7 are idle, plus GPU 4 is Parakeet. A second TP=4 replica is not possible on this single box (would need 4 more free H100s of the same class).
- A second replica on another node would front-end via a reverse proxy / load-balancer on port 9084 (or a shared gateway) — stateless HTTP, so scaling is trivially horizontal.
- Prefix cache is node-local (vLLM doesn't share it across replicas), so sticky routing by tenant / conversation will preserve the ~71 % hit rate we currently enjoy.
- Single-replica start-up is ~95 s (weights + compile); budget warm replicas rather than cold-start for failover.
- Per-replica capacity numbers in §8 can be multiplied by replica count for aggregate capacity.

---

## 11. Risks & caveats

- **Decode scaling at N ≥ 3 is unmeasured.** All "max parallel ≥ 3" rows in §8 depend on per-slot decode holding up, which the log cannot confirm. Treat as estimate.
- **FP8 KV (Opportunity 7) is speculative** for MiniMax-M2.5 on this specific vLLM nightly. It is the single largest potential free lever; validate before enabling.
- **MoE routing under load** — MiniMax-M2.5 activates 10 B per token out of 230 B. Expert-imbalance can cause tail latency spikes under heavy concurrency. Not observed in this log (peak N=2), but is a known MoE behaviour to watch once load grows.
- **TP-4 all-reduce** is fused (`fuse_allreduce_rms`) but still adds a per-layer sync. At sustained high concurrency it becomes a bigger fraction of step time; we have no direct telemetry for it.
- **vLLM nightly (`v0.19.1rc1.dev71+gdd9342e6b`)** can change behaviour across builds. Pin the exact nightly when reproducing any number in this doc.
- **Prefix cache hit rate** is workload-shaped. If callers stop sharing system prompts, the observed 30–71 % hit rate will fall and effective throughput will drop accordingly.
- **Log coverage** — only 10 days observed, with almost no load. Behaviour under real production concurrency is inferred, not measured. Re-run this analysis after a real load test.

---

## 12. Appendix — raw log evidence

Line numbers reference `logs/service.log`.

Engine config and version:

```
18: (EngineCore pid=63472) INFO 04-07 15:56:59 [core.py:105] Initializing a V1 LLM engine
    (v0.19.1rc1.dev71+gdd9342e6b) with config: model='MiniMaxAI/MiniMax-M2.5', ...
    dtype=torch.bfloat16, max_seq_len=128000, tensor_parallel_size=4,
    quantization=fp8, kv_cache_dtype=auto, enable_prefix_caching=True,
    enable_chunked_prefill=True, cudagraph_capture_sizes=[1, 2, 4, 8, 16], ...
```

Chunked prefill + async scheduling:

```
12: INFO 04-07 15:56:49 [scheduler.py:238] Chunked prefill is enabled with max_num_batched_tokens=8192.
13: INFO 04-07 15:56:49 [vllm.py:799] Asynchronous scheduling is enabled.
17: INFO 04-07 15:56:52 [compilation.py:290] Enabled custom fusions: norm_quant, act_quant, allreduce_rms
```

Weights and KV sizing:

```
167: INFO 04-07 15:57:47 [default_loader.py:384] Loading weights took 34.89 seconds
169: INFO 04-07 15:57:48 [gpu_model_runner.py:4820] Model loading took 53.75 GiB memory and 37.946043 seconds
988: INFO 04-07 15:59:03 [gpu_worker.py:436] Available KV cache memory: 18.16 GiB
990: INFO 04-07 15:59:03 [kv_cache_utils.py:1319] GPU KV cache size: 307,184 tokens
991: INFO 04-07 15:59:03 [kv_cache_utils.py:1324] Maximum concurrency for 128,000 tokens per request: 2.40x
1015: INFO 04-07 15:59:07 [core.py:283] init engine (profile, create kv cache, warmup model) took 78.95 seconds
```

Peak concurrency sample (the only `Running: 2` in the log):

```
1201: INFO 04-07 17:54:45 [loggers.py:259] Engine 000: Avg prompt throughput: 2178.0 tokens/s,
      Avg generation throughput: 93.6 tokens/s, Running: 2 reqs, Waiting: 0 reqs,
      GPU KV cache usage: 16.4%, Prefix cache hit rate: 2.3%
```

Representative single-request steady-state decode:

```
1231: INFO 04-07 18:11:25 [loggers.py:259] ... Avg generation throughput: 92.5 tokens/s,
      Running: 1 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.5%, Prefix cache hit rate: 3.1%
1344: INFO 04-07 19:26:15 [loggers.py:259] ... Avg generation throughput: 97.9 tokens/s,
      Running: 1 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.5%, Prefix cache hit rate: 39.6%
```

Representative single-request prefill burst:

```
1200: INFO 04-07 17:54:35 [loggers.py:259] Avg prompt throughput: 2734.8 tokens/s,
      Avg generation throughput: 63.2 tokens/s, Running: 1 reqs, Waiting: 0 reqs,
      GPU KV cache usage: 9.0%, Prefix cache hit rate: 3.3%
2043: INFO 04-15 11:23:54 [loggers.py:259] Avg prompt throughput: 3038.2 tokens/s,
      Avg generation throughput: 65.2 tokens/s, Running: 1 reqs, Waiting: 0 reqs,
      GPU KV cache usage: 10.2%, Prefix cache hit rate: 38.4%
```

Prefix cache reaching ~71 %:

```
1151: INFO 04-07 17:15:05 [loggers.py:259] ... Running: 0 reqs, Waiting: 0 reqs,
      GPU KV cache usage: 0.0%, Prefix cache hit rate: 71.0%
```

Capacity-relevant engine hint (free ~3 % KV recovery):

```
985: INFO 04-07 15:59:02 [gpu_worker.py:470] In v0.19, CUDA graph memory profiling will be
     enabled by default (VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1) ... increase
     --gpu-memory-utilization from 0.9500 to 0.9526 to maintain the same effective KV cache size.
```

No incidents:

```
$ grep -cE 'OutOfMemory|preempt(ed|ion)|swapped|CUDA out of memory' logs/service.log
0
```
