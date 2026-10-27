# Jade — the power-flexibility envelope of disaggregated LLM inference

DSCI-650 (High Performance Data Science, RIT, Fall 2026) project — Karthik Mattu.

**Question.** When a grid operator asks an AI datacenter to cut power, how much can a live LLM inference
service give up, how fast, for how long, at what latency cost per customer tier, and what happens when the
limit lifts? Jade measures this per **pool** of a prefill/decode-disaggregated vLLM deployment (prefill
GPU, decode GPU) and per **lever** (admission by tier, drain, GPU power cap, SM clock lock), and reports it
as a "flexibility datasheet" in the units a utility contract uses (depth, speed, duration, cost, release).

## How the system is put together

```
  jade_loadgen.py / jade_e3.py  (open-loop Poisson load, 3 tiers; event driver)
          |  HTTP, X-Jade-Tier header
          v
  jade_admission_proxy.py :8300  (the lever: reject / defer / drain by tier; logs every command)
          |
          v
  vLLM toy proxy :8192  (routes each request: prefill first, then decode)
      |                          |
      v                          v
  vLLM prefill :8100  --KV cache over NIXL/UCX-->  vLLM decode :8200
  (GPU 0, compute-bound)                            (GPU 1, memory-bound)
      ^                          ^
      +---- jade_power_sampler.py (NVML, 20 Hz, both GPUs, one clock) ----+
```
Work is split by **phase** (pipeline parallelism across two GPUs): the prefill GPU builds a request's KV
cache, which is transferred to the decode GPU, which generates the answer. Every log carries wall-clock
timestamps from the same node, so power, commands and requests line up without a synchronisation step.

## Code layout

| Path | What it is |
|---|---|
| `disagg/jade_power_sampler.py` | Per-GPU power, SM clock, throttle flags at 20 Hz via NVML → CSV (streams to disk) |
| `disagg/jade_loadgen.py` | Open-loop Poisson load generator (aiohttp): Azure-trace request lengths, three tiers, latency timed from the *scheduled* send time; E1 rate sweep |
| `disagg/jade_admission_proxy.py` | The admission lever: HTTP proxy that rejects (429) or defers requests of chosen tiers while an event is active; `/control` endpoint; command log |
| `disagg/jade_e3.py` | E3 event driver: baseline → command → hold → release → after, at constant load |
| `disagg/jade_e3_rows.py` | Turns power + event + request logs into datasheet rows (definitions in `schema/README.md`) |
| `disagg/jade_smoke_plot.py` | Per-phase × per-GPU power table and time-series plot |
| `disagg/disagg_smoke_client.py` | Smoke-test client: TTFT disaggregated vs single GPU, phased |
| `disagg/jade_disagg_smoke.sh` | Slurm job: two-GPU smoke test + E1 sweep |
| `disagg/jade_e3.sh` | Slurm job: E3 events on the disaggregated pair |
| `disagg/jade_mono.sh` | Slurm job: monolithic baseline (one GPU, E1's exact sweep) |
| `disagg/logs_*/` | Raw results of each run (job id in the name) |
| `energy_probe.py`, `jade_e0.sh` | E0/E0.1: single-GPU joules per prefill vs decode token |
| `cloud/lambda_cap_probe.py`, `cloud/lambda_preflight.sh` | Verifies a GPU power cap and clock lock really take effect (rented Lambda A100, root) |
| `schema/` | The lever-curve schema shared with the sister cloud project (v0.3; row definitions in README) |
| `tests/` | No-GPU end-to-end tests: mock vLLM server, synthetic power with known answers |
| `ai_planning/` | Plans written with AI assistance and the refined versions (see `AI_USAGE.md`) |
| `proposal/`, `docs/` | Proposal source and design notes |

Planned, not yet in the repository: the C++ instrument (`jade-probe`: NVML sampler thread + ring buffer;
`jade-load`: libcurl, thread per stream) and the MPI harness (rank 0 controller, one rank per GPU,
`Isend` commands, `Barrier`-aligned windows, `Gather` of samples). The Python versions above are the
prototype the C++ code must reproduce.

## Setup

Python 3.10+. The analysis tools and tests need only `aiohttp` (and optionally `matplotlib` for plots):
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install aiohttp matplotlib
```
GPU runs (RIT SPORC, 2 × A100-PCIe-40GB) additionally need `vllm==0.28.0`, `nixl`, `nvidia-ml-py`, and vLLM's
`toy_proxy_server.py` from the v0.28.0 tag (`tests/v1/kv_connector/nixl_integration/`) in `disagg/`.
Nothing is compiled; "building" is installing these packages.

## Run

**Without a GPU (≈1 min)** — end-to-end check of load generator, admission proxy, event driver and row analysis:
```bash
bash tests/run_mock_tests.sh        # prints PASS/FAIL per check; exit code = number of failures
```

**On SPORC (Slurm)** — from `~/jade/disagg` with the Azure trace staged in `~/jade/traces/`:
```bash
sbatch jade_disagg_smoke.sh                       # smoke test + E1 rate sweep (2 GPUs, ~30 min)
sbatch jade_mono.sh                               # monolithic baseline (1 GPU, ~20 min)
E3_RATES="1 2" E3_EVENTS="shed_reject drain shed_defer" E3_GAP_S=90 sbatch jade_e3.sh   # E3 events
python jade_e3_rows.py --dir logs_e3_<jobid>      # datasheet rows
```
Azure trace: `https://raw.githubusercontent.com/Azure/AzurePublicDataset/master/data/AzureLLMInferenceTrace_conv.csv`.

## Results so far (details in the run folders)
- **E0/E0.1** (1 GPU): decode costs ~25× more energy per token than prefill; the stock 250 W cap binds prefill, not decode.
- **Smoke + E1** (`logs_21928556`): two-pool serving works; at sustainable load the **decode pool holds most of the power** (4 req/s: decode 204 W, prefill 62 W).
- **Monolithic baseline** (`logs_mono_21956668`): one GPU sustains 16 req/s; the disaggregated pair saturates at 4–8 req/s. The KV handoff runs at ~0.28 GB/s (≈ 69 ms + 0.1 ms per prompt token) — a host-staged transfer path, to be fixed in E2.
- **E3-lite** (`logs_e3_21956669`): drain removes ~0.6 of nominal power from the decode pool in 2–4 s; shedding the flex tier is shallow (≤ 0.12); a deferred backlog spikes the prefill GPU to its limit on release.
- **Root levers** (Lambda A100-SXM): power cap and clock lock take effect in ~0.2–0.3 s; a decode-like load (~209 W of 400 W) is only capped below its own draw.
