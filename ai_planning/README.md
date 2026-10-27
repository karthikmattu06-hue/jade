# AI planning record

Plans drafted with Claude and the versions I refined, in order. Decisions are mine; the files show how the plan changed.

| Date | File | What changed and why |
|---|---|---|
| 7 Aug 2026 | `01_brief_v1_2026-08-07.md` | **Initial plan**: energy-aware scheduling over llm-d on Kubernetes; fallback ladder down to one GPU. |
| 11 Sep 2026 | (in `02`) | Syllabus check: the course is C++/threads/MPI/CUDA, so the Python tools became the *prototype* and a C++/MPI instrument the course deliverable. |
| 17–21 Sep 2026 | `../proposal/` | Proposal: software-only power controls per phase. Go/no-go 2-GPU smoke test passed 21 Sep. Graded 75/100 (scope sound; writing read as AI-generated, too much unexplained terminology). |
| 9 Oct 2026 | (in `02`) | Literature pass (PowerSlider, PALS, Yan et al.): node-level power control of inference is published, and "no root access" has no real customer — the facility operator who is paid for demand response has root. I rejected the proposal's gap. |
| 19 Oct 2026 | `02_brief_v2_2026-10-19.md` | **Refined plan**: the flexibility envelope (depth, speed, duration, cost by tier, release) per pool and per lever, including root levers (power cap, clock lock). |
| 21 Oct 2026 | (in `02`) | RIT Research Computing refused GPU power-control access; GCP quota denied. Root levers moved to a rented Lambda A100, verified working. |
| 26 Oct 2026 | — | Monolithic baseline shows disaggregation costs ~25 % energy on this small model. Decision: disaggregation is the deployed reality, so it is the setting; the monolithic run is a baseline, and the study is how much power to take from each pool. |
