# cloud/ — root-lever verification on a rented GPU

RIT SPORC gives no root, so GPU power caps and clock locks are measured on a rented VM (Lambda Cloud, 1 × A100-SXM4-40GB, root).

- `lambda_cap_probe.py` — loads one GPU (compute-bound matmul ≈ prefill, or memory-bound copy ≈ decode), steps the power cap down a ladder and back, then locks and unlocks the SM clock, sampling NVML at 20 Hz. Verdict is behavioural: mean power in the second half of each hold must sit at the cap.
- `lambda_preflight.sh` — the paid test hour: hardware report, sudo check, both probes, reset.

Result (21 Oct 2026): cap and clock lock both take effect; a binding cap settles in ~0.2–0.3 s; the memory-bound load draws ~209 W on the 400 W card, so caps of 320/240 W do not bind it, 180/140 W do.

## Multi-GPU sessions (added 10–11 Nov 2026)

- `lambda_dev_setup.sh` — one-command setup of a blank Lambda VM: Python 3.12 venv via uv (vLLM 0.28's flashinfer crashes on the image's 3.10), vLLM + NIXL, vLLM toy proxy, Azure trace, model weights.
- `lambda_kv_diag.sh` — one prefill/decode pair: hand-off tax, 4/8/16 req/s sweep, UCX transport tables, vLLM KV-transfer metrics. Result on 4× A6000 (10 Nov): transfer mostly ≤ 5 ms over `cuda_ipc` zero-copy (SPORC: median 156 ms); pair served 16 req/s. TCP is needed for UCX's control path (forbidding it stops the server).
- `lambda_session.sh` — the 8× A100 production session: four pairs at once (`check`, `up`, `interference`, `e1`, `levers`, `admission`, `rows`, `down`). Levers step: pair 0 caps the prefill GPU, pair 1 the decode GPU, pairs 2–3 lock the SM clock on prefill / decode. Admission step: shed reject / defer / ramped defer / drain, four replicates.
