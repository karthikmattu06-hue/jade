# AI usage

**Tool.** Claude (Anthropic), through the Claude desktop app with access to this project folder, used
throughout the project since August 2026. No other code-generation tools.

## What it was used for
- **Planning:** experiment design (E0–E3), the datasheet framing, the schema, the risk/fallback plan. Initial
  and refined plans are in `ai_planning/`. Decisions in those plans were mine; several of Claude's proposals
  were changed or rejected after discussion (e.g. the research gap was re-chosen on 19 Oct after I judged the
  original "software-only levers" premise to have no real user; the monolithic run was kept as a baseline
  rather than as a reason to doubt disaggregation, 26 Oct).
- **Literature search and checking:** finding papers and checking claims against them. Every reference used in
  the proposal was exported by me from DBLP/ACM/IEEE, not taken from the chatbot.
- **Code:** see below.
- **Slides:** layout and wording help; the content is my work and I can explain every slide.

## Which code was mostly AI-generated
Mostly AI-generated (first versions written by Claude to my specification, then reviewed and modified by me):
`disagg/jade_power_sampler.py`, `jade_loadgen.py`, `jade_admission_proxy.py`, `jade_e3.py`, `jade_e3_rows.py`,
`jade_smoke_plot.py`, `disagg_smoke_client.py`, all `*.sh` Slurm scripts, `energy_probe.py`,
`cloud/lambda_cap_probe.py`, `cloud/lambda_preflight.sh`, everything in `tests/`.
Not AI-generated: run decisions, node/partition choices, what to measure and why.

## How I checked it
- **Mock end-to-end tests** (`tests/run_mock_tests.sh`): load generator, admission proxy, event driver and row
  analysis run against a fake vLLM server; the row analysis is checked against synthetic power with known answers.
- **Synthetic-data checks caught real bugs:** a trailing moving average made every measured "speed" ~0.5–0.8 s
  too slow (fixed to a centred 0.5 s window); the schema's recovery test is met instantly by shallow levers
  (diagnostic column added); results were written only at the end of a run, so a Slurm timeout lost them (now
  streamed to disk).
- **Cross-checks on real runs:** decode never recomputed prefill (vLLM metrics: prompt throughput 0, external
  cache hit 100 %); GPU-to-role mapping verified (only the decode GPU rises when it serves alone); E0.1's
  per-token energies predict E1's per-pool power within ~10–15 %; the E1 pattern matches a published H200
  measurement (Yan et al., arXiv:2609.24639).
- **Behavioural verification of root levers:** the cap probe checks measured power under load, not just the
  command's return code.
- **Reading line by line:** every file was read and explained to me at the point of use before I ran it; I asked
  for and checked line-level explanations where I did not follow.

Kept up to date at each update.
