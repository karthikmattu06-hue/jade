# cloud/ — root-lever verification on a rented GPU

RIT SPORC gives no root, so GPU power caps and clock locks are measured on a rented VM (Lambda Cloud, 1 × A100-SXM4-40GB, root).

- `lambda_cap_probe.py` — loads one GPU (compute-bound matmul ≈ prefill, or memory-bound copy ≈ decode), steps the power cap down a ladder and back, then locks and unlocks the SM clock, sampling NVML at 20 Hz. Verdict is behavioural: mean power in the second half of each hold must sit at the cap.
- `lambda_preflight.sh` — the paid test hour: hardware report, sudo check, both probes, reset.

Result (21 Oct 2026): cap and clock lock both take effect; a binding cap settles in ~0.2–0.3 s; the memory-bound load draws ~209 W on the 400 W card, so caps of 320/240 W do not bind it, 180/140 W do.
