# Context doc — Jade Control Plane (Modern Cloud Computing, Fall 2026)

**Portable context.** Paste into a fresh chat when working on the Modern Cloud Computing project. Written 9 Sep 2026 (week 3 of 14) after the first measurements from the sister project landed. Owner: Karthik Mattu, MS AI, RIT, graduating May 2027. Companion doc: the Jade brief (`brief-v1-2026-08-07.md`) and the programme review (`programme-review-2026-09-09.md`) in the Jade project.

---

## 1. What this is, in one paragraph

A semester project for **Modern Cloud Computing** (Prof. Fang). Solo. The deliverable is a **Kubernetes operator on GKE that enforces a declared power budget on a live LLM serving deployment** by pulling levers the serving software already exposes — admission control, draining, replica count — while an SLO alert protects premium users. Prefill (prompt-reading) and decode (answer-writing) run on **separate pools**, and the operator chooses **which pool** to cut; that choice is the control surface disaggregated serving creates, and the reason this project exists alongside its measurement twin. It is a thermostat: you declare the temperature (a power budget), it reads the thermometer (GPU telemetry from DCGM), it nudges the furnace (the vLLM cluster) until the two agree. Grid events arrive on Pub/Sub; the whole thing is provisioned with Terraform and torn down between sessions.

## 2. The one question, and the analogy that makes the numbers make sense

Grids pay large users to shed load on request. Emerald AI has shown AI datacenters can do it for training by capping GPU power. Interactive inference is harder — caps need root, and users are waiting. **Can a software-only control plane hit a power budget on a live LLM service, fast, without breaking the latency promise to premium users?**

Think of a GPU as a car with the engine running:

| State | What the GPU is doing | Measured on an A100 (250 W limit) | In the analogy |
|---|---|---|---|
| **Parked** | model loaded, zero requests | ≈ 40 W | engine idling |
| **Decode** | writing the answer, one token at a time | ≈ 200 W | traffic jam — light load, barely moving |
| **Prefill** | reading the whole prompt at once | ≈ 230 W | highway — high power, huge distance |

Two ways to measure fuel: **watts** (gallons per hour — what the grid sees) and **joules per token** (gallons per mile — what the bill sees). Decode costs ~27× more per token than prefill, yet prefill draws more power, because it processes ~30× more tokens per second. Both are true; keep them apart.

## 3. Relationship to the sister project (Jade, DSCI-650) — read this before building anything

Jade measures, on RIT's A100 cluster under Slurm, what each lever does: watts shed, ramp time, recovery, latency cost. **This project builds the hand that pulls the levers.** Measurement and control; two halves of one programme (the same split Emerald runs between its field measurements and its Conductor product).

**What is shared:** the problem domain; the vLLM serving engine; the Qwen2.5-1.5B model; NVIDIA GPU power telemetry as the measured quantity; and **one JSON file** — the lever-curve schema in §9. Nothing else. No code, no compute, no experiments, no graded artifact. The operator runs on synthetic curves from day one and is complete without Jade's results.

**Disclosure.** Both professors (Travis for DSCI-650, Fang for this course) are being told the overlap in week 3, in these terms, with the offer to adjust the split. Do not let the overlap grow quietly: if a design decision would make this project depend on Jade's code or compute, stop and flag it.

**Hardware mismatch, by design.** Jade's curves come from a 250 W A100 (parked ≈ 40 W). This project runs a 70 W T4 (parked ≈ 12 W, expected). Absolute watts do not transfer; **fractions of nominal do**. Every curve the operator consumes is expressed as a fraction of the GPU's nominal (limit) power. The T4 demonstrates the mechanism, not the magnitude — say so in the report rather than being asked.

## 4. What Jade has measured so far (9 Sep 2026) — the facts this design rests on

1. **Draining is the lever.** Busy 197–230 W, parked 39–43 W. Stop sending a GPU work and let in-flight answers finish, and ~80% of its draw is gone in seconds. Nothing else comes close.
2. **The power limit binds on prefill, not decode — measured by the driver's own throttle flag.** SW_POWER_CAP set ~90% of the time in prefill-heavy windows, 0% in small-batch decode. So a GPU power cap is a real lever for prompt-reading and a no-op for answer-writing.
3. **Mean power under the limit is not headroom.** A prefill window *averaging* 230 W under a 250 W limit was capped 90% of the time — the cap acts on 260–350 W peaks. **The operator must read the throttle-reason flag (`DCGM_FI_DEV_CLOCK_THROTTLE_REASONS`), not compute `limit − mean`.**
4. **Batching is an efficiency lever, not a power lever.** Joules per decode token fall 25× from batch 1 to 32 while mean power *rises* 177 → 214 W. A batch cap sheds ~37 W at most; it is admission control by another name. Not a real-time lever here.
5. **Sleep mode is a memory lever.** A replica nobody is sending requests to is already at ~40 W. `POST /sleep` frees GPU memory and buys the last few watts. It holds the floor; it does not create it. Stretch, not headline.
6. **Parked power drifts with temperature** (~0.4 W/°C). Use the hot value as the floor.
7. **Disaggregation gives every lever a target.** On a split cluster, draining the prefill pool is the deeper cut per GPU (~230 → 40 W) and costs only TTFT; draining the decode pool is shallower (~200 → 40 W) and costs TPOT if forced. Admission is the prefill-pool intake valve by construction. A power cap (if anyone had root) would be a prefill-only lever. Rebound on recovery hits the prefill pool. Jade's E2 (weeks 6–8, two A100s) measures the per-pool curves; until then they are estimated from E0.1's prefill- and decode-dominated windows.

## 5. Architecture

### Components

| Component | Kind | What it is | Here |
|---|---|---|---|
| Grid event replay | write | 100-line script: reads ERCOT/CAISO price CSV (gridstatus), publishes an event when a threshold is crossed | Exercises the loop with realistic shapes: 30-min shed, 2-h ramp, false alarm |
| Pub/Sub topic `dr-events` | configure | GCP managed message bus | Decouples grid side from cluster side; events queue if the operator restarts |
| `PowerBudget` CRD | write | Custom resource: "target ≤ 60% of nominal from 14:00–14:30; protect premium p99 ≤ 2 s" | The declared desired state; creating one starts an event, expiry starts recovery |
| `PowerBudget` operator | write (Python, kopf) | Reconcile loop every few seconds: read target, read watts + throttle flag + premium p99 from Prometheus, pick cheapest lever that closes the gap, act, log | The brain. Backs off a lever when the SLO burn-rate alert fires. Ramp-limited recovery |
| Policy ConfigMap | write (format) | The lever curves, §9 schema, fractions of nominal | Separates policy from mechanism; swapping synthetic → measured is one `kubectl apply` |
| Admission proxy | write (~60 lines) | HTTP proxy in front of vLLM; reads a ConfigMap; returns 429 to flex-tier requests when the budget is active | The mechanism the Gateway Inference Extension would provide, without its alpha-API risk |
| vLLM Deployments × 2 | configure | `prefill` and `decode` pools, each its own Deployment with a role label; Qwen2.5-1.5B fp16, `/metrics` on, pinned image | Replica count is a lever **per pool**. Live T4 runs one pool; the simulation runs both |
| Load generator | write | Tiered trace (premium/flex), replayed identically per policy, per-request latency to CSV | Adapt vLLM's bench tool or k6 |
| DCGM exporter | configure (GKE-managed) | Per-GPU watts, energy, util, temperature, **throttle reasons** | The thermometer |
| Managed Prometheus | configure | Scrapes DCGM and vLLM `/metrics`; PromQL for the operator | The logbook |
| Grafana + SLO | configure | One dashboard; burn-rate alert on premium p99 | The user-protection guardrail and the demo surface |
| Terraform | provision | GKE zonal, CPU pool 2 × e2-medium Spot, GPU pool 1 × T4 Spot, Pub/Sub, IAM, budget alert, destroy target | The cluster idles at $0 when destroyed |

### Levers, ordered by measured effect — every one carries a pool target

Disaggregation is not a lever you pull; it is what turns "shed" into "shed *which pool*". The operator's real decision is **pool × lever**, chosen by watts recovered per unit of SLO cost.

| Lever | Pool | Depth | Speed | Costs the user | Scope |
|---|---|---|---|---|---|
| Admission by tier | prefill intake, by construction | up to ~80% of active draw × flex share | seconds | flex waits / 429 | must |
| Drain to parked | prefill first (deeper, TTFT only); decode second (shallower, TPOT) | ~80% of active draw per GPU | seconds (in-flight work finishes) | queueing | must |
| Replica count | per pool | one replica's share | 1–2 min | higher latency | must |
| Sleep / wake | per replica | last few watts below parked | seconds | wake latency | stretch |
| Batch cap | decode | ~small; efficiency lever | minutes, restart | throughput | no |
| GPU power cap / clocks | prefill only (measured) | — | — | — | out of scope (root; Emerald's lever, deliberately not ours) |

### Evaluation — the same operator, twice

1. **Live T4** — proves the wiring end to end: Pub/Sub → CRD → reconcile → proxy/Deployment → DCGM → Grafana. Small watts, real mechanism.
2. **Simulated cluster, two pools** — prefill and decode replicas that obey Jade's measured per-pool A100 curves (fraction-of-nominal), same operator code, compared against a **monolithic baseline** (one pool, blended curve) under the same operator. This is where the quantitative claims come from: depth reached vs. target, time to reach it, premium p99 held or not, rebound on recovery, with and without hysteresis — and whether cutting prefill or decode first holds the SLO at a given budget.

The controller is the contribution: pool × lever selection under an SLO with noisy telemetry, hysteresis so it does not oscillate, a rebound guard on recovery. The disaggregated-vs-monolithic comparison is the thesis in the cloud project's own terms.

## 6. Plan

**Must · weeks 3–9 · ≈ 60 h — this alone is the deliverable**
Paid billing + T4 quota + confirmed credit; Terraform (cluster, pools, Pub/Sub, IAM, budget alert, destroy); one vLLM replica; managed DCGM → Managed Prometheus → one Grafana board; `PowerBudget` CRD + kopf operator; **pool-aware model — `prefill` and `decode` Deployments, every lever action carries a pool target, reconcile picks pool × lever**; admission proxy + replica-count levers; tiered load generator; synthetic per-pool curves in the §9 schema; CI on GitHub Actions.

**Should · weeks 10–12 · ≈ 30 h — the graduate-level evidence**
Pub/Sub event replay from ERCOT prices; SLO burn-rate rule → operator back-off; load Jade's curves (v0 week 6 from a one-GPU load sweep, v1 week 11 from the two-GPU event-replay experiment, per pool); **two-pool simulated cluster vs monolithic baseline, same operator**; ramp-limited recovery.

**Stretch · week 13+ · ≈ 30 h — none of it changes the grade**
A second T4 Spot running vLLM disaggregated prefill for real (one node, NCCL KV connector; Turing GPUs lack FlashAttention 2, so xformers/Triton — fragile); Gateway API Inference Extension replacing the proxy; sleep/wake; ArgoCD.

## 7. Week-1 blockers (verified 9 Sep 2026)

- **GPUs are not available on a GCP free trial.** Google's terms: adding GPUs is restricted while the billing account is a non-billable trial. Upgrade to paid billing first, then request `GPUS_ALL_REGIONS` quota (defaults to 0; days; sometimes refused on new accounts). The $300 trial credit expires 90 days after activation — shorter than the semester — so use the $50 course credit first and activate the trial only when the GPU pool is ready to spend it.
- **Multiple vLLM replicas on one T4 need explicit GPU sharing.** T4 has no MIG; GKE time-sharing/MPS must be set on the node pool (`gke-gpu-sharing-strategy`, `gke-max-shared-clients-per-gpu`) and neither enforces memory limits. Must-list uses one replica.
- **`InferenceObjective` (Gateway API Inference Extension) is alpha** with breaking changes expected and undocumented shedding semantics. Start with the proxy; verify the extension by hand before depending on it.
- **Rubric unknown.** This plan assumes Prof. Fang grades on architecture, managed-service breadth and a working demo. Get the rubric in week 3; the must-list is the first thing to re-cut if it differs.

## 8. Calendar (Mondays, approximate to RIT)

| wk | date | this project | interface |
|---|---|---|---|
| 3 | Sep 7 | unblock: billing, quota, rubric; disclosure email to both professors | both profs |
| 4 | Sep 14 | Terraform foundation | **schema frozen** |
| 5 | Sep 21 | GKE up, vLLM up, DCGM in Prometheus | |
| 6 | Sep 28 | CRD + operator skeleton | curves v0 arrive (E1) |
| 7 | Oct 5 | admission proxy, replica lever | |
| 8 | Oct 12 | load gen, synthetic curves, first end-to-end event by hand | |
| 9 | Oct 19 | Grafana, loop closed | |
| 10 | Oct 26 | Pub/Sub replay; SLO burn-rate → back-off | |
| 11 | Nov 2 | load A100 curves; simulated-cluster harness | curves v1 arrive (E3) |
| 12 | Nov 9 | evaluate: live T4 + simulated; rebound guard | |
| 13 | Nov 16 | evaluate, write | |
| 14 | Nov 23 | report, demo, teardown | Thanksgiving |

## 9. Lever-curve schema v0.2 — the one shared artifact

Frozen week 4. Both projects read and write exactly this. Values are **fractions of the GPU's nominal (limit) power** so an A100 curve can drive a T4 policy. Every lever carries `pool` (`prefill` | `decode` | `any`); the same lever name may appear once per pool. `source.topology` says whether the file came from a monolithic or a disaggregated run. `measured` vs `estimated` is tracked per field so nobody quotes an estimate as data.

```json
{
  "schema_version": "0.2",
  "source": {
    "project": "jade",
    "experiment": "E0.1",
    "date": "2026-09-09",
    "gpu": "NVIDIA A100-PCIE-40GB",
    "nominal_w": 250,
    "model": "Qwen/Qwen2.5-1.5B-Instruct",
    "serving": "vLLM 0.28.0",
    "batch": 8,
    "topology": "monolithic",
    "topology_note": "one GPU serving both phases; E2 (two A100s, disaggregated) will emit topology='disaggregated' with per-pool states measured rather than estimated"
  },
  "states": {
    "parked_frac":              {"value": 0.17, "status": "measured", "note": "43 W hot; 39 W cold; ~0.4 W/C"},
    "decode_pool_busy_frac":    {"value": 0.79, "status": "estimated", "note": "from the decode-dominated window, 197 W; E2 measures on its own GPU"},
    "prefill_pool_busy_frac":   {"value": 0.92, "status": "estimated", "note": "from the prefill-dominated window, 230 W, cap flag 90%; E2 measures on its own GPU"},
    "monolithic_busy_frac":     {"value": 0.84, "status": "measured", "note": "mixed traffic, ~210 W"}
  },
  "levers": [
    {
      "name": "admission_by_tier",
      "pool": "prefill",
      "authority": "operator",
      "depth_frac_per_unit_flex_share": {"value": 0.62, "status": "estimated", "note": "= busy - parked on the prefill pool, scaled by flex share; E3 measures"},
      "ramp_s":       {"value": 8,    "status": "estimated", "note": "prefill queue empties in seconds; E1 measures"},
      "recovery_s":   {"value": 30,   "status": "estimated", "note": "ramp-limited re-admission; E3 measures"},
      "rebound_frac": {"value": 0.10, "status": "estimated", "note": "backlog hits the prefill pool at TDP; E3 measures"},
      "delta_ttft_premium_ms": {"value": 0,    "status": "estimated"},
      "delta_tpot_premium_ms": {"value": 0,    "status": "estimated"},
      "delta_p99_flex_ms":     {"value": null, "status": "n/a", "note": "flex is deferred, not slowed"}
    },
    {
      "name": "drain_to_parked",
      "pool": "prefill",
      "authority": "operator",
      "depth_frac":   {"value": 0.76, "status": "estimated", "note": "230 -> 40 W = 191 W on the prefill-dominated window"},
      "ramp_s":       {"value": 3,    "status": "estimated", "note": "prefill jobs are sub-second"},
      "recovery_s":   {"value": 15,   "status": "estimated"},
      "delta_ttft_premium_ms": {"value": null, "status": "n/a", "note": "new requests wait for the pool to reopen"},
      "delta_tpot_premium_ms": {"value": 0,    "status": "estimated", "note": "in-flight answers keep streaming from decode"}
    },
    {
      "name": "drain_to_parked",
      "pool": "decode",
      "authority": "operator",
      "depth_frac":   {"value": 0.63, "status": "estimated", "note": "197 -> 40 W = 158 W on the decode-dominated window"},
      "ramp_s":       {"value": 12,   "status": "estimated", "note": "output_len x TPOT for in-flight answers to finish; E1 measures"},
      "recovery_s":   {"value": 15,   "status": "estimated"},
      "delta_ttft_premium_ms": {"value": 0,    "status": "estimated"},
      "delta_tpot_premium_ms": {"value": null, "status": "n/a", "note": "forcing it stalls in-flight answers; normally never forced"}
    },
    {
      "name": "replica_count",
      "pool": "any",
      "authority": "operator",
      "depth_frac_per_replica": {"value": 0.63, "status": "estimated", "note": "same as drain for that pool's GPU; the GPU parks, the node stays"},
      "ramp_s":       {"value": 90,   "status": "estimated", "note": "pod termination + drain"},
      "recovery_s":   {"value": 120,  "status": "estimated", "note": "model reload"},
      "delta_p99_premium_ms": {"value": null, "status": "estimated", "note": "depends on remaining capacity vs load; E1 curve"}
    },
    {
      "name": "batch_cap",
      "pool": "decode",
      "authority": "operator",
      "depth_frac":   {"value": 0.15, "status": "measured", "note": "177 -> 214 W across batch 1..32; efficiency lever"},
      "ramp_s":       {"value": 120,  "status": "estimated", "note": "restart-only flag"},
      "role": "efficiency, not power"
    },
    {
      "name": "sleep_wake",
      "pool": "any",
      "authority": "operator",
      "depth_frac":   {"value": 0.05, "status": "estimated", "note": "parked 43 W -> ~30 W"},
      "ramp_s":       {"value": 3,    "status": "estimated"},
      "recovery_s":   {"value": 5,    "status": "estimated"}
    },
    {
      "name": "gpu_power_cap",
      "pool": "prefill",
      "authority": "root",
      "scope": "out of scope for both projects",
      "depth_frac":   {"value": 0.08, "status": "measured", "note": "already binding 90% of prefill samples at 250 W; settable floor 150 W; no-op on decode above ~200 W"}
    }
  ]
}
```

The operator treats every `estimated` field as replaceable and logs which version of the file, and which pool × lever, each decision used.

## 10. Standing constraints and decisions (do not relitigate)

- **Software levers only.** No root on any node; no `nvidia-smi -pl`, no clock locks. This is the design, not a limitation.
- **No quality-degrading levers.** Reasoning effort, quantization and similar knobs belong to the model provider and the end user, not the datacenter. The authority filter: a lever exists only if someone has the right to pull it.
- **GCP course credits ($50) belong to this course.** Jade gets no cloud spend. Confirm the exact credit amount; the plan's cost model is ≈ $0.20–0.29/h with everything Spot and the cluster destroyed between sessions.
- **Python with kopf** for the operator (Karthik is rebuilding Python fluency and will read every line). Go/kubebuilder is the industry default and a second language mid-semester.
- **One replica in the must-list.** Multi-replica GPU sharing is stretch.
- **Read the throttle flag**, not `limit − mean`.
- **Curves are fractions of nominal**, never absolute watts.
- **Pool-aware from day one.** Two Deployments with role labels even while only one runs live; every lever carries `pool`. A phase-blind operator cannot consume Jade's E2 output and is not the control half of this programme.

## 11. Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| GPU quota refused or slow | medium | File this week. Fallback tier: whole control plane on CPU-only vLLM or a curve-driven mock — grade intact, watts lost |
| Trial credits expire before finals | certain if activated now | Course credit first; activate the trial when the GPU pool is ready |
| Spot T4 preempted mid-demo | medium | Record the demo; simulated-cluster evaluation is the primary evidence |
| Total hours (Jade + this + RL course + RA + Intutic + job search) | high | Cap at the must-list until Jade's two-GPU experiment is done; stretch items are deletable |
| Professors object to the shared substrate | low if disclosed now | Week-3 email; different questions, different clusters, one data schema |
| Gateway extension unstable | medium | Proxy first; extension is stretch |

## 12. About Karthik, for a fresh chat

MS AI at RIT (grad May 2027), IIT Delhi EE + ICT, ~6 years industry (government audit, product engineering, AI operations lead), co-founder of Intutic. Targeting AI Product Manager roles; Emerald AI (Dr. Ayse Coskun, Chief Scientist, who advised him in Feb 2026 to learn vLLM, Ray, llm-d and GPU energy work) is the target. Prior work: CoordDR (sole author; MILP coordinator for geographic datacenter load shifting; 87–96% lower QoS cost at equal curtailment), ercot-rtcb-bench (open benchmark for battery bidding under ERCOT RTC+B). Jade is "his Emerald" — the umbrella for his grid-flexible-AI-datacenter programme; the name is deliberate.

**How he wants to work:** propose a plan and validate it before building; ask clarifying questions before diving deep; push back when reasoning is inconsistent; complete outputs in one block; analogies for hard concepts (the car/thermostat ones above are his); code with detailed comments for a first-time reader of the K8s/cloud stack; concepts explained at the point of use; token-efficient. When a research or design idea is proposed, review it thoroughly (prior art + numbers) before he adopts it, and state claims at their true size.

## 13. References

- Emerald AI field paper, Williams et al., arXiv 2606.25098 (June 2026) — inference SLO boundaries listed as open
- Acun et al., BU PEAC Lab, ACM e-Energy '26 — FlexDC-Sim, request-level mechanisms and prefill/decode listed as future work
- Khanal, Roh, Dvorkin, arXiv 2608.19622 — firm/flexible/interruptible tiers need depth, duration, ramp, recovery per event
- GCP Free Program terms (GPUs excluded on trial): https://docs.cloud.google.com/free/docs/free-cloud-features
- GKE GPU sharing strategies: https://docs.cloud.google.com/kubernetes-engine/docs/concepts/timesharing-gpus
- Gateway API Inference Extension, `InferenceObjective` (alpha): https://gateway-api-inference-extension.sigs.k8s.io/api-types/inferenceobjective/
- vLLM sleep mode needs `--enable-sleep-mode` and `VLLM_SERVER_DEV_MODE=1`; pin the version
- kopf: https://kopf.readthedocs.io/
