# Lever-curve schema (the one artifact shared between Jade and the Jade Control Plane)

- `lever-curves-v0.2.json` - frozen v0.2 contract (Sep 2026), E0.1 numbers. Unchanged.
- `lever-curves-v0.3.json` - **0.3 rev 2, frozen for the control plane on 19 Oct 2026** (after cloud-side review M1-M4).
  Structure (names, nesting, units, definitions, enums) is frozen; values may be filled or replaced and entries appended.
  Any structural change bumps `revision` and is reviewed by both chats first. The operator's loader checks
  `schema_version` + `revision` and refuses anything unexpected.
- `README-v0.2.md` - the previous README.

The schema is a contract (field names and meanings). No code is shared between the projects:
each side writes its own producer/consumer.

## Rules (carried over from v0.2)
- Power values are fractions of the GPU's nominal (limit) power, so an A100 curve can drive a T4 policy.
- Every lever carries `pool` (`prefill` | `decode` | `any`).
- Every numeric field is `{"value", "status", "note"?}`; status is `measured` | `estimated` | `assumed` | `n/a`.
  Never quote an estimate or assumption as data.

## What changed in v0.3
1. **Five datasheet rows per lever entry**, matching Jade brief v2: `depth_frac`, `speed_s`, `duration_s`, `cost`, `release`.
   v0.2's `ramp_s` -> `speed_s`; `recovery_s` and `rebound_frac` move under `release`.
2. **A lever entry is (name, pool, setting).** The same lever can appear several times with different settings
   (cap 0.9/0.8/0.7/0.6; admission reject vs defer). `at_load` records the offered load the entry was measured at,
   because depth depends on how busy the pool was.
3. **`authority`: `operator` | `root`.** Root levers `gpu_power_cap` and `gpu_clock_lock` added per pool; v0.2's
   "out of scope for both projects" note dropped.
4. **`source_hardware`** per entry, pointing into a `hardware` table: GPU, nominal W, model, serving stack,
   topology, site, experiment, date, and `origin` = `measured_jade` | `published`. A published number also needs
   a citation. This is how an operator tells an A100-PCIe measurement from an H200 number taken from a paper.
5. **`tiers`**: names, per-tier SLOs (`assumed`, from QoServe, ASPLOS '26), default mix 1/3 each with sensitivity mixes.
6. **`event_protocol`**: the window every row is computed from.
7. **`states`** is now a list per pool, each with its `source_hardware`.
8. (rev 2, M1) **`depth_frac` is the depth at the entry's own `at_load.tier_mix`**, not per unit of shed share.
   Admission at the default mix: 0.62 x 1/3 = 0.21. The per-unit figure is kept as `depth_per_unit_shed_share`.
9. (rev 2, M2) **`composition`**: one lever per pool per event. Exclusive on one GPU: drain with anything; cap with
   clock lock. Combined depth is not measured; an operator may assume at most min(sum, busy - parked) (`estimated`).
   Depths do not simply add: admission lowers load, so a cap may stop binding.
10. (rev 2, M3) **Duration beyond the hold.** `duration_s` carries `bound`: `null` when an SLO broke during the hold
    (the value is the time it broke), `"lower"` when none broke (value = hold length, true duration is longer).
    Defer entries get one extra long-hold event (600 s), because deferred flex is capped by its 600 s TTLT SLO;
    Jade's admission proxy `--max-hold-s` must be >= 600 for that event.
11. (rev 2, M4) **`interpolation`**: key = `at_load.pool_busy_frac`; target 3 loads per entry, minimum 2.
    Update I (29 Oct) will have one load per entry.
12. (rev 2) The tier **header name is project-local** (Jade `X-Jade-Tier`, control plane `X-Service-Tier`);
    tier names and reject/defer semantics stay shared.

## Row definitions (both sides compute them exactly this way)
One event: `baseline_s` at constant offered load -> command at t_cmd -> `hold_s` -> release at t_rel -> `after_s`.
P = the pool's GPU power (the instantaneous NVML field if the driver has it; else the ~1 s averaged reading, noted).
P_base = mean P over the last 30 s of baseline. P_hold = mean P over the second half of the hold.

| Row | Field(s) | Definition |
|---|---|---|
| Depth | `depth_frac` | (P_base - P_hold) / nominal_w |
| Speed | `speed_s` | t90 - t_cmd, where t90 = first time P has covered 90 % of the drop from P_base to P_hold |
| Duration | `duration_s`, `bound` | time from t_cmd until the first served tier leaves its SLO (`bound: null`); if none does, the hold length with `bound: "lower"`; `n/a` if the lever builds no queue |
| Cost | `cost.<tier>.d_ttft_p99_ms`, `d_tpot_p99_ms`, `served_frac`, `d_ttlt_p99_ms` | p99 during the hold minus p99 during baseline, per tier; `served_frac` = completed / offered during the hold |
| Release | `release.recovery_s` | t_rec - t_rel, where t_rec = first time P and premium p99 TTFT (10 s window) are both within 10 % of baseline |
| | `release.rebound_frac` | (max P in the after window - P_base) / nominal_w; 0 if never above baseline |
| | `release.rebound_s` | time P spends above P_base + 0.05 x nominal in the after window |
| | `release.d_ttft_p99_after_ms` | premium p99 TTFT in the after window minus baseline |

Latency percentiles count failed or cut requests as +infinity (so they cannot hide). Requests shed by design
(`reject`) count only in `served_frac`, not in latency. The cost key `all` means every tier together.

**Transfer assumption.** Fractions of nominal let an A100 curve drive a T4 policy, but that transfer is an
*assumption*, not a measured result: different GPUs have different idle floors and cap behaviour. On the control
plane the T4 is a wiring test only; its numbers come from the simulator.

## Who fills what
- **Jade** fills `measured` entries from E1 (states, at_load) and E3 (every lever row), on `sporc_a100_pcie_disagg`
  and, for root levers, on SPORC-with-RC or a rented A100-PCIe.
- **The control plane** may fill empty entries from published curves (`origin: "published"`, with citation) so its
  simulator can run before Jade's numbers exist, and swaps them for Jade's when they land. It never edits Jade's
  `measured` entries.
