# Reference shortlist for the DSCI-650 proposal (verified 17 Sep 2026; DBLP BibTeX exported the same day)

**Status:** `proposal/mattu_proposal.bib` now holds the DBLP-exported entries for all 23 cited references (pulled through Karthik's browser from dblp.org, keys renamed, DOI links kept). Radovanović was dropped — DBLP only carries its arXiv version and IEEE Trans. Power Systems is outside DBLP. DBLP records BiScale (arXiv v1 title) for the paper now called DualScale; the proposal text says BiScale to match. DBLP's VoltanaLLM title carries an encoding artifact ("Control Íand"); the .bib uses the publisher's title.

Rules from the instructions: ≥10 refs (solo), ≥75 % peer-reviewed, ≥4 from 2021+, ≥2 from the primary venue, DOIs where they exist, BibTeX exported from DBLP / ACM DL / IEEE Xplore — **not** hand-written. Export each entry from the DBLP link, then rename the key to the one in the first column so the .tex compiles.

"Verified" = I confirmed title, venue, year (and DOI where given) against Crossref, the publisher page, or the USENIX page today. "Check" = one field I could not confirm from here.

## Theme A — Disaggregated and phase-aware LLM serving (the system under study)

| key | Reference | Status | Role in the proposal |
|---|---|---|---|
| `kwon2023vllm` | Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention," SOSP '23. DOI 10.1145/3600006.3613165 | verified | The engine; foundational |
| `zhong2024distserve` | Zhong et al., "DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving," OSDI '24, pp. 193–210 (no DOI; USENIX) | verified | Origin of P/D disaggregation for goodput |
| `patel2024splitwise` | Patel et al., "Splitwise: Efficient Generative LLM Inference Using Phase Splitting," ISCA '24. DOI 10.1109/ISCA59077.2024.00019 | verified | Origin of the per-phase power asymmetry; Azure traces |
| `qin2025mooncake` | Qin et al., "Mooncake: Trading More Storage for Less Computation — A KVCache-centric Architecture for Serving LLM Chatbot," FAST '25, pp. 155–170 (no DOI) | verified | Disaggregation in production; KV transfer as the communication cost |
| `agrawal2024sarathi` | Agrawal et al., "Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve," OSDI '24, pp. 117–134 (no DOI) | verified | The non-disaggregated alternative (chunked prefill) — the monolithic baseline's design |

## Theme B — Power and energy management for LLM inference (the levers)

| key | Reference | Status | Role |
|---|---|---|---|
| `patel2024polca` | Patel et al., "Characterizing Power Management Opportunities for LLMs in the Cloud," ASPLOS '24. DOI 10.1145/3620666.3651329 | verified | Power capping/oversubscription for LLM clusters; the privileged-lever baseline |
| `stojkovic2025dynamollm` | Stojkovic et al., "DynamoLLM: Designing LLM Inference Clusters for Performance and Energy Efficiency," HPCA '25. DOI 10.1109/HPCA61900.2025.00102 | verified | Energy-aware cluster reconfiguration; steady-state, not events |
| `qiu2024userve` | Qiu et al., "Power-aware Deep Learning Model Serving with μ-Serve," USENIX ATC '24, pp. 75–93 (no DOI) | verified | Frequency scaling + multiplexing under SLOs |
| `kakolyris2024slo` | Kakolyris et al., "SLO-Aware GPU DVFS for Energy-Efficient LLM Inference Serving," IEEE Computer Architecture Letters, 2024. DOI 10.1109/LCA.2024.3406038 | verified (note exact title) | Per-phase DVFS; privileged lever |
| `yu2026voltana` | Yu et al., "VoltanaLLM: Energy-Efficient and SLO-Aware Disaggregated LLM Serving via Adaptive Frequency Control and State-Space Routing," ISC High Performance 2026. DOI 10.23919/ISC.2026.11520495 | verified via Crossref; check authors on Xplore | **Closest peer-reviewed work**: disaggregated + DVFS + routing. Steady-state energy; no exogenous event; requires clock control |
| `you2023zeus` | You, Chung, Chowdhury, "Zeus: Understanding and Optimizing GPU Energy Consumption of DNN Training," NSDI '23, pp. 119–139 (no DOI) | verified (first author is **You**, not Chung) | GPU energy measurement methodology; power/time trade-off |
| *preprint* `dualscale2026` | "DualScale: Energy-Efficient Disaggregated LLM Serving via Phase-Aware Placement and DVFS," arXiv:2602.18755 | arXiv | Nearest scheduler work; cite, does not count |
| *preprint* `rapid2026` | "Power Aware Dynamic Reallocation For Inference" (RAPID), arXiv:2601.12241 | arXiv | Power budgets over disaggregated inference; cite, does not count |
| *preprint* `illusion2026` | "The Illusion of Power Capping in LLM Decode," arXiv:2605.11999 | arXiv | Caps don't bind on decode — corroborates E0.1; cite, does not count |

## Theme C — Datacenter power capping and demand response (who consumes the output)

| key | Reference | Status | Role |
|---|---|---|---|
| `wu2016dynamo` | Wu et al., "Dynamo: Facebook's Data Center-Wide Power Management System," ISCA '16. DOI 10.1109/ISCA.2016.48 | verified | Foundational DC power capping |
| `li2020thunderbolt` | Li et al., "Thunderbolt: Throughput-Optimized, Quality-of-Service-Aware Power Capping at Scale," OSDI '20, pp. 1241–1255 (no DOI) | verified | Priority-aware capping — the "tiers" idea at Google scale |
| `liu2013dcdr` | Liu et al., "Data center demand response: avoiding the coincident peak via workload shifting and local generation," SIGMETRICS '13. DOI 10.1145/2465529.2465740 | verified | Foundational DC demand response |
| `radovanovic2023carbon` | Radovanović et al., "Carbon-Aware Computing for Datacenters," IEEE Trans. Power Systems 38(2):1270–, 2023. DOI 10.1109/TPWRS.2022.3173250 | venue/vol verified; **check DOI on Xplore** | Google's grid-aware DC flexibility |
| `acun2026flexdc` | Acun, Hankendi, Levine, Reynolds, Bardwick, Coskun, "Investigating Power Consumption Flexibility of AI Data Centers for Demand Response Participation," ACM e-Energy '26. DOI 10.1145/3744255.3798112 | verified | Measurement-first flexibility study (batch jobs); gap: no serving stack, no phases |
| *preprint* `williams2026powerflex` | Williams et al., "Power-Flexible AI Data Centers," arXiv:2606.25098 | arXiv | Field demo; the inference-SLO gap they declare |
| *preprint* `khanal2026shift` | Khanal et al., "Shift or curtail? …," arXiv:2608.19622 | arXiv | Defines depth/duration/ramp/recovery per event — our output format |

## Theme D — Measurement methodology and the target venue (ICPE)

| key | Reference | Status | Role |
|---|---|---|---|
| `cavagna2026sweetspot` | Pizzini Cavagna et al., "SweetSpot: An Analytical Model for Predicting Energy Efficiency of LLM Inference," ICPE '26. DOI 10.1145/3777884.3797011 | verified | **ICPE paper 1** — analytical energy model our E1 curves can be compared against |
| `wallace2025icpe` | Wallace, Ombuki-Berman, Ezzati-Jivan, "Optimization Strategies for Enhancing Resource Efficiency in Transformers & Large Language Models," ICPE '25 (short). DOI 10.1145/3676151.3719379 | verified | **ICPE paper 2** |
| `samsi2023words` | Samsi et al., "From Words to Watts: Benchmarking the Energy Costs of Large Language Model Inference," IEEE HPEC '23. DOI 10.1109/HPEC58863.2023.10363447 | verified | Early LLM inference energy benchmark |
| `niu2026tokenpowerbench` | Niu et al., "TokenPowerBench: Benchmarking the Power Consumption of LLM Inference," AAAI '26, 40(38):32582–32590. DOI 10.1609/aaai.v40i38.40535 | verified | Phase-aligned power attribution; methodology |

## Counts

Cited in the proposal: 23 — 18 peer-reviewed (78 %), 5 preprints; 2021+: 20; ICPE: 2. (Samsi and Radovanović are in this list but not cited.) A 2-page proposal will comfortably cite 15–20 of these; if trimming, drop `agrawal2024sarathi`, `samsi2023words`, `radovanovic2023carbon` first.

## Venue

Primary: **ICPE** (ACM/SPEC Int'l Conf. on Performance Engineering) — rank B on the instructor's list; takes characterisation and measurement studies; two related papers above. Backup: **CCGrid** (B) or **IPDPS** (A). Not HPDC: its only related item is a 2025 poster. Not HotCarbon (workshop, disqualified). Not e-Energy (not on the list; rank unverified; relevance is the instructor's call).

## DBLP export

Search each title at https://dblp.org/search?q= and use "export → BibTeX (condensed)". For the USENIX papers DBLP gives `@inproceedings` with pages and no DOI — that's correct. For the arXiv preprints DBLP gives `@article{... journal={CoRR}}`; keep them as arXiv entries with the identifier.
