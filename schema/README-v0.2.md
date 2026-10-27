# Lever-curve schema (the one artifact shared between Jade and the Jade Control Plane)

`lever-curves-v0.2.json` is the frozen v0.2 contract (week 4, Sep 2026), populated with Jade's E0.1 numbers (one A100, monolithic). Source of record: `docs/control-plane-context.md` §9.

Rules both projects follow:
- Values are fractions of the GPU's nominal (limit) power, so an A100 curve can drive a T4 policy.
- Every lever carries `pool` (`prefill` | `decode` | `any`); a lever name may appear once per pool.
- Every numeric field carries `status`: `measured` | `estimated` | `n/a`. Never quote an estimate as data.
- `source.topology` is `monolithic` or `disaggregated`.

Pending change (agreed 14 Oct 2026, not yet applied): the `gpu_power_cap` lever is marked `"scope": "out of scope for both projects"`. Jade may now measure cap and clock levers on rented GCP A100s. v0.3 should (a) drop that scope note, (b) add `"authority": "root"` levers `gpu_power_cap` and `gpu_clock_lock` per pool with the same fields as the others, and (c) add a per-lever `"source_hardware"` so an operator can tell an A100-measured curve from an H200 number taken from a paper. Neither side writes code against v0.3 until both chats have seen it.
