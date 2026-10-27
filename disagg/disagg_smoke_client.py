#!/usr/bin/env python3
"""
disagg_smoke_client.py v2 -- what does the KV handoff cost, and what do both
GPUs draw while it happens?

v2 (19 Oct 2026) adds, on top of v1's TTFT comparison:
  * a PHASED run so the power log shows parked -> busy -> parked on both pools:
        idle_pre  (30 s, model loaded, no requests)   -> parked baseline per GPU
        disagg    (N requests via the proxy)          -> both GPUs work
        gap       (10 s idle)                         -> lets the 1 s power average settle
        mono      (N requests straight to decode GPU) -> ONLY the decode GPU works
        idle_post (30 s idle)                         -> parked again (now warmer)
  * two CSVs on the SAME wall clock (time.time()) as jade_power_sampler.py:
        phases.csv   : phase, t_start, t_end
        requests.csv : one row per request -- send / first-token / end times
  The printed TTFT lines are unchanged from v1, so the result compares
  directly with the 21 Sep run (75.1 ms disagg vs 41.3 ms mono).

Built-in sanity check on the GPU mapping: in the "mono" phase only GPU 1
(decode) should rise above parked. If GPU 0 rises instead, the role labels
are swapped -- check CUDA_DEVICE_ORDER in the job script.

TTFT = wall time from sending the request to the first streamed chunk that
carries text. Proxy path: prefill on GPU 0 + KV transfer + first decode step
on GPU 1. Mono path: prefill on GPU 1 alone. Requests are sequential (one at a
time), so this is a latency floor with no queueing -- not a throughput number.

Usage:
  python disagg_smoke_client.py --model Qwen/Qwen2.5-1.5B-Instruct --n 20 \
      --proxy localhost:8192 --mono localhost:8200 --out-dir logs_<job>
"""
import argparse, csv, json, os, statistics, time, urllib.request

PROMPT = ("Explain, in careful detail, why a GPU running the decode phase of a "
          "language model is limited by memory bandwidth rather than arithmetic. ") * 6


def one_request(host: str, model: str, max_tokens: int = 64) -> dict:
    """One streaming completion. Returns wall-clock stamps plus durations.

    Durations use perf_counter (monotonic, precise); the wall stamps use
    time.time() so they can be lined up with the power CSV."""
    body = json.dumps({"model": model, "prompt": PROMPT, "max_tokens": max_tokens,
                       "temperature": 0, "stream": True}).encode()
    req = urllib.request.Request(f"http://{host}/v1/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    w0 = time.time(); t0 = time.perf_counter(); ttft = None; ok = True
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            for raw in r:                               # server-sent events: "data: {...}" lines
                line = raw.decode().strip()
                if not line.startswith("data:") or line == "data: [DONE]":
                    continue
                chunk = json.loads(line[5:])
                if ttft is None and chunk["choices"][0].get("text"):
                    ttft = time.perf_counter() - t0    # first chunk with actual text
    except Exception as err:                            # record the failure, keep the run going
        print(f"  request to {host} failed: {err}")
        ok = False
    total = time.perf_counter() - t0
    ttft = ttft if ttft is not None else float("nan")
    return {"t_send": w0, "t_first": w0 + ttft, "t_end": w0 + total,
            "ttft_s": ttft, "total_s": total, "ok": int(ok)}


def idle(seconds: float) -> None:
    """Send nothing. The GPUs sit with the model loaded -- the 'parked' state."""
    time.sleep(seconds)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--proxy", required=True, help="host:port of toy_proxy_server (disaggregated path)")
    ap.add_argument("--mono", required=True, help="host:port of the decode instance (monolithic path)")
    ap.add_argument("--idle-s", type=float, default=30.0, help="parked phase before and after")
    ap.add_argument("--gap-s", type=float, default=10.0, help="idle between the two busy phases")
    ap.add_argument("--out-dir", default=".", help="where phases.csv / requests.csv go")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    for host in (a.proxy, a.mono):            # one warm-up each, discarded, BEFORE the baseline
        one_request(host, a.model)

    phases, rows = [], []

    def phase(name, fn):
        t0 = time.time(); fn(); phases.append((name, t0, time.time()))
        print(f"  phase {name:9s} {phases[-1][2] - t0:6.1f} s", flush=True)

    def busy(label, host):
        def run():
            for i in range(a.n):
                r = one_request(host, a.model); r.update(path=label, i=i); rows.append(r)
        return run

    phase("idle_pre", lambda: idle(a.idle_s))
    phase("disagg", busy("disagg", a.proxy))
    phase("gap", lambda: idle(a.gap_s))
    phase("mono", busy("mono", a.mono))
    phase("idle_post", lambda: idle(a.idle_s))

    with open(os.path.join(a.out_dir, "phases.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["phase", "t_start", "t_end"])
        for name, t0, t1 in phases:
            w.writerow([name, f"{t0:.4f}", f"{t1:.4f}"])
    with open(os.path.join(a.out_dir, "requests.csv"), "w", newline="") as f:
        cols = ["path", "i", "t_send", "t_first", "t_end", "ttft_s", "total_s", "ok"]
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in rows:
            w.writerow({k: (f"{r[k]:.4f}" if isinstance(r[k], float) else r[k]) for k in cols})

    # Same printout as v1, for a like-for-like comparison with 21 Sep.
    med = {}
    for label, name in (("disagg", "disaggregated (proxy)"), ("mono", "monolithic (decode GPU only)")):
        ttfts = [r["ttft_s"] for r in rows if r["path"] == label and r["ok"]]
        totals = [r["total_s"] for r in rows if r["path"] == label and r["ok"]]
        if not ttfts:
            print(f"{name:30s} no successful requests"); continue
        med[label] = statistics.median(ttfts)
        print(f"{name:30s} TTFT median {med[label]*1000:7.1f} ms  "
              f"p90 {sorted(ttfts)[max(0, int(0.9*len(ttfts))-1)]*1000:7.1f} ms   "
              f"total median {statistics.median(totals)*1000:7.1f} ms   (n={len(ttfts)})")
    if len(med) == 2:
        print(f"\nKV-handoff tax on this prompt (median TTFT difference): {(med['disagg']-med['mono'])*1000:.1f} ms")
        print("21 Sep reference (skl-a-54, eager): 75.1 vs 41.3 ms -> +33.8 ms")
    print(f"wrote {a.out_dir}/phases.csv and {a.out_dir}/requests.csv")


if __name__ == "__main__":
    main()
