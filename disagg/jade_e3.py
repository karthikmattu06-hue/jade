#!/usr/bin/env python3
"""
jade_e3.py -- E3-lite: grid events against live disaggregated serving.

WHAT ONE EVENT IS (schema v0.3 event_protocol)
    baseline 60 s  ->  COMMAND  ->  hold 120 s  ->  RELEASE  ->  after 60 s
Throughout, open-loop Poisson load arrives at a constant rate (same generator
and request shapes as E1: jade_loadgen.py is imported, not copied). The command
and release go to jade_admission_proxy.py's /control endpoint; the proxy logs
them with timestamps. The power sampler (separate process) records both GPUs.
From these three logs the analysis computes the five datasheet rows: depth,
speed, duration, cost per tier, release.

THE EVENTS (per load level)
    shed_reject  -- flex tier refused (429) during the hold. Work lost, no rebound expected.
    shed_defer   -- flex tier held at the proxy, released all at once at the end.
                    The burst is the rebound the "release" row measures.
    drain        -- every tier refused: new work stops, in-flight finishes, both pools park.
E1 predicts: shedding 1/3 of requests cuts decode power only a little (decode
draws ~190 W with even one request in flight), while a drain cuts ~165 W on
decode. E3 measures whether that holds.

OUTPUT (wall clock, same as the sampler)
    e3_events.csv   : one row per event -- rate, kind, t_start, t_cmd, t_rel, t_end
    e3_phases.csv   : <event>_base / _hold / _after / _tail / _gap windows (for jade_smoke_plot.py)
    e3_requests.csv : one row per request, as E1, plus event id and the proxy decision visible as status

Usage:
  python jade_e3.py --proxy localhost:8300 --model Qwen/Qwen2.5-1.5B-Instruct \
      --trace ~/jade/traces/AzureLLMInferenceTrace_conv.csv --rates 2 3 4 --out-dir logs_<job>
"""
import argparse, asyncio, csv, os, sys, time

import aiohttp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from jade_loadgen import Clock, RowSink, load_trace, make_specs, one_request, pct, write_phases  # noqa: E402

REQ_COLS = ["event", "kind", "rate", "i", "tier", "prompt_tokens", "out_req", "out_got", "t_sched", "t_send",
            "t_first", "t_end", "lag_ms", "ttft_ms", "tpot_ms", "status", "err"]

EVENTS = {
    "shed_reject": {"shed": ["flex"], "mode": "reject"},
    "shed_defer":  {"shed": ["flex"], "mode": "defer"},
    "drain":       {"shed": ["premium", "standard", "flex"], "mode": "reject"},
}


async def control(session, proxy, body):
    async with session.post(f"http://{proxy}/control", json=body) as r:
        return await r.json()


async def run_event(session, a, ev_id, rate, kind, trace, rng, clk, rows, phases, events):
    url = f"http://{a.proxy}/v1/completions"
    total = a.base_s + a.hold_s + a.after_s
    specs, _ = make_specs(rate, total, 0.0, rng, trace, a.tier_mix, a.max_prompt, a.max_out, 100, 20000)
    t0 = time.perf_counter() + 1.0                    # schedule built first, then anchored
    for s in specs:
        s["t_sched"] += t0
    ev_rows = []
    tasks = [asyncio.create_task(one_request(session, url, a.model, s, clk, ev_rows, a.timeout_s))
             for s in specs]
    t_cmd, t_rel = t0 + a.base_s, t0 + a.base_s + a.hold_s

    await asyncio.sleep(max(0.0, t_cmd - time.perf_counter()))
    await control(session, a.proxy, dict(EVENTS[kind], note=f"{ev_id} start {kind} @ {rate:g} rps"))
    m_cmd = time.perf_counter()
    await asyncio.sleep(max(0.0, t_rel - time.perf_counter()))
    await control(session, a.proxy, {"shed": [], "note": f"{ev_id} release"})
    m_rel = time.perf_counter()
    await asyncio.sleep(max(0.0, t0 + total - time.perf_counter()))
    m_end = time.perf_counter()

    done, pending = await asyncio.wait(tasks, timeout=a.tail_s)   # in-flight finish
    for p in pending:
        p.cancel()
    await asyncio.gather(*pending, return_exceptions=True)
    m_tail = time.perf_counter()
    await asyncio.sleep(a.gap_s)                                  # let power settle before the next event
    m_gap = time.perf_counter()

    for r in ev_rows:
        r["event"] = ev_id; r["kind"] = kind
        rows.append(r)                         # RowSink: on disk as soon as the event ends
    w = clk.wall
    events.append({"event": ev_id, "rate": rate, "kind": kind, "t_start": f"{w(t0):.4f}",
                   "t_cmd": f"{w(m_cmd):.4f}", "t_rel": f"{w(m_rel):.4f}", "t_end": f"{w(m_end):.4f}"})
    for name, x0, x1 in (("base", t0, m_cmd), ("hold", m_cmd, m_rel), ("after", m_rel, m_end),
                         ("tail", m_end, m_tail), ("gap", m_tail, m_gap)):
        phases.append((f"{ev_id}_{name}", w(x0), w(x1)))

    # One-line console summary per event: premium p99 TTFT before / during / after, flex outcome.
    def p99(tier, lo, hi):
        xs = [float(r["ttft_ms"]) if r["ttft_ms"] else float("inf") for r in ev_rows
              if r["tier"] == tier and lo <= float(r["t_sched"]) < hi and r["status"] != "http429"]
        return pct(xs, 99)
    sb, sc, sr, se = w(t0), w(m_cmd), w(m_rel), w(m_end)
    flex = [r for r in ev_rows if r["tier"] == "flex" and sc <= float(r["t_sched"]) < sr]
    print(f"  {ev_id} {kind:11s} {rate:g} rps | premium p99 TTFT base {p99('premium', sb, sc):8.0f} "
          f"hold {p99('premium', sc, sr):8.0f} after {p99('premium', sr, se):8.0f} ms | flex in hold: "
          f"{sum(r['status'] == 'ok' for r in flex)}/{len(flex)} served | {len(pending)} cut", flush=True)


async def main_async(a):
    import random
    rng = random.Random(a.seed)
    trace = load_trace(os.path.expanduser(a.trace))
    os.makedirs(a.out_dir, exist_ok=True)
    clk = Clock(); phases, events = [], []
    rows = RowSink(os.path.join(a.out_dir, "e3_requests.csv"), REQ_COLS)

    def save_small():
        """events + phases are tiny: rewrite both after every event, so a timeout keeps them."""
        write_phases(os.path.join(a.out_dir, "e3_phases.csv"), phases)
        if events:
            with open(os.path.join(a.out_dir, "e3_events.csv"), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(events[0])); w.writeheader(); w.writerows(events)
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=0)) as session:
        await control(session, a.proxy, {"shed": [], "mode": "reject", "note": "e3 start: clear"})
        # warm-up, discarded
        warm, _ = make_specs(1.0, 5.0, time.perf_counter(), rng, trace, a.tier_mix, a.max_prompt, a.max_out, 100, 20000)
        await asyncio.gather(*(one_request(session, f"http://{a.proxy}/v1/completions", a.model, s, clk, [], a.timeout_s)
                               for s in warm[:5]), return_exceptions=True)
        m = time.perf_counter(); await asyncio.sleep(a.idle_s)
        phases.append(("idle_pre", clk.wall(m), clk.wall(time.perf_counter())))
        n = 0
        for rate in a.rates:
            for kind in a.events:
                n += 1
                await run_event(session, a, f"ev{n:02d}", rate, kind, trace, rng, clk, rows, phases, events)
                save_small()
        m = time.perf_counter(); await asyncio.sleep(a.idle_s)
        phases.append(("idle_post", clk.wall(m), clk.wall(time.perf_counter())))

    save_small(); rows.close()
    print(f"wrote {a.out_dir}/e3_events.csv, e3_phases.csv, e3_requests.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proxy", required=True, help="host:port of jade_admission_proxy.py")
    ap.add_argument("--model", required=True)
    ap.add_argument("--trace", required=True)
    ap.add_argument("--rates", type=float, nargs="+", default=[2, 3, 4])
    ap.add_argument("--events", nargs="+", default=list(EVENTS), choices=list(EVENTS))
    ap.add_argument("--base-s", type=float, default=60.0)
    ap.add_argument("--hold-s", type=float, default=120.0)
    ap.add_argument("--after-s", type=float, default=60.0)
    ap.add_argument("--tail-s", type=float, default=60.0)
    ap.add_argument("--gap-s", type=float, default=30.0)
    ap.add_argument("--idle-s", type=float, default=30.0)
    ap.add_argument("--tier-mix", type=float, nargs=3, default=[1, 1, 1])
    ap.add_argument("--max-prompt", type=int, default=4096)
    ap.add_argument("--max-out", type=int, default=512)
    ap.add_argument("--timeout-s", type=float, default=300.0)
    ap.add_argument("--seed", type=int, default=650)
    ap.add_argument("--out-dir", default=".")
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
