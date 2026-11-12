#!/usr/bin/env python3
"""
jade_loadgen.py -- E1 open-loop load generator: power and latency vs request rate.

WHAT E1 ANSWERS
  How do per-pool watts and per-tier latency change as offered load rises?
  That curve is the "depth" baseline every lever is measured against: you can
  only shed watts that the load is drawing in the first place.

OPEN LOOP (the one idea to understand before reading the code)
  The smoke test was CLOSED loop: send, wait for the answer, send the next.
  A slow server then slows the client down too, so latency looks better than
  real users would see -- the "coordinated omission" problem. Real users arrive
  on their own schedule. So here every request's send time is fixed IN ADVANCE
  from a Poisson process (exponential gaps, mean 1/rate), and the request goes
  out at that time whether or not earlier ones have finished. Latency is
  measured from the SCHEDULED time (t_sched), not from when we managed to send.
  t_send - t_sched ("lag") is logged: if it grows, the generator itself is
  overloaded and the run is invalid at that rate.

REQUEST SHAPES
  Prompt and output lengths are drawn from the Azure LLM inference trace
  (conversation set, Patel et al. "Splitwise", ISCA 2024 -- columns
  TIMESTAMP, ContextTokens, GeneratedTokens). Only the LENGTHS are used, not the
  timestamps; the rate is ours. The prompt is sent as random token ids (so
  prefix caching can never hit) and ignore_eos=True makes the model generate
  exactly the drawn number of tokens. Long tails are clipped (see --max-prompt,
  --max-out); the clipped fraction is printed.

TIERS
  Each request is tagged premium / standard / flex. Default mix is equal thirds,
  following QoServe (Goel et al., ASPLOS 2026), which assumes an equal mix and
  tests 70-15-15 and 15-15-70 as sensitivity; no production tier mix is
  published. In E1 there is no lever, so tiers are served identically -- the
  tag matters from E3 on, when the admission proxy sheds flex first. The tag
  travels as the X-Jade-Tier header, which that proxy will read.

OUTPUT (same wall clock as jade_power_sampler.py)
  e1_phases.csv   : idle_pre, then per rate "rate_<r>" (arrival window) and
                    "tail_<r>" (in-flight requests finishing), then idle_post
  e1_requests.csv : one row per request (tier, lengths, times, status)

Usage:
  python jade_loadgen.py --url localhost:8192 --model Qwen/Qwen2.5-1.5B-Instruct \
      --trace ~/jade/traces/AzureLLMInferenceTrace_conv.csv --out-dir logs_<job>
"""
import argparse, asyncio, csv, math, os, random, statistics, time, json

import aiohttp  # installed with vLLM

TIERS = ("premium", "standard", "flex")


def load_trace(path):
    """(prompt_tokens, output_tokens) pairs from the Azure trace."""
    out = []
    with open(path) as f:
        for r in csv.DictReader(f):
            p, o = int(r["ContextTokens"]), int(r["GeneratedTokens"])
            if p >= 1 and o >= 1:
                out.append((p, o))
    return out


def pct(xs, q):
    """q-th percentile; failed/cut requests enter as +inf so they cannot hide."""
    if not xs:
        return float("nan")
    xs = sorted(xs)
    k = min(len(xs) - 1, max(0, math.ceil(q / 100 * len(xs)) - 1))
    return xs[k]


REQ_COLS = ["rate", "i", "tier", "prompt_tokens", "out_req", "out_got", "t_sched", "t_send",
            "t_first", "t_end", "lag_ms", "ttft_ms", "tpot_ms", "status", "err"]


def raise_fd_limit():
    """Lift this process's open-file limit to the hard maximum.
    Every in-flight or HELD request is an open socket. Linux's default soft
    limit is 1024, and on 11 Nov a defer event at 24 req/s held ~960 requests
    at once: the client and proxy ran out of sockets ("Too many open files")
    and three of four pairs crashed. Raising the soft limit needs no root."""
    import resource
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    want = hard if hard != resource.RLIM_INFINITY else 1_048_576
    if soft < want:
        resource.setrlimit(resource.RLIMIT_NOFILE, (want, hard))


class RowSink(list):
    """A list that ALSO streams every appended row to a CSV file as it arrives.

    Why: the first version wrote the CSV only at the very end, so a job killed by
    Slurm's time limit lost every request it had measured. Now each finished
    request is flushed to disk as it finishes, so the file is complete up to the moment
    the job dies. It still behaves as a normal list for the summary."""
    def __init__(self, path, cols):
        super().__init__()
        self._f = open(path, "w", newline="")
        self._w = csv.DictWriter(self._f, fieldnames=cols, extrasaction="ignore")
        self._w.writeheader(); self._f.flush()

    def append(self, row):
        super().append(row)
        self._w.writerow(row)
        self._f.flush()          # every row: a hard kill (Slurm SIGKILL) loses at most the one in flight

    def close(self):
        self._f.flush(); self._f.close()


def write_phases(path, phases):
    """Rewrite the (small) phases file after every phase, for the same reason."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["phase", "t_start", "t_end"])
        for n, t0, t1 in phases:
            w.writerow([n, f"{t0:.4f}", f"{t1:.4f}"])


class Clock:
    """Monotonic clock for scheduling, converted to wall time for the CSVs."""
    def __init__(self):
        self.w0, self.m0 = time.time(), time.perf_counter()
    def wall(self, mono):
        return self.w0 + (mono - self.m0)


async def one_request(session, url, model, spec, clk, rows, timeout_s):
    """Wait until spec['t_sched'] (monotonic), send, stream, record one row.
    The row is written in `finally`, so a request cut at the end of a tail
    (asyncio.CancelledError) is still recorded -- as status 'cut'."""
    delay = spec["t_sched"] - time.perf_counter()
    if delay > 0:
        await asyncio.sleep(delay)
    t_send = time.perf_counter(); t_first = None; n_out = None; n_chunks = 0
    status = "error"; err = ""
    body = {"model": model, "prompt": spec["ids"], "max_tokens": spec["out"],
            "ignore_eos": True, "temperature": 0, "stream": True,
            "stream_options": {"include_usage": True}}
    try:
        async with session.post(url, json=body, headers={"X-Jade-Tier": spec["tier"]},
                                timeout=aiohttp.ClientTimeout(total=timeout_s)) as resp:
            if resp.status != 200:
                status = f"http{resp.status}"; err = (await resp.text())[:200]
            else:
                async for raw in resp.content:            # server-sent events, one line each
                    line = raw.decode().strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    chunk = json.loads(data)
                    if chunk.get("usage"):                 # last chunk: exact token count
                        n_out = chunk["usage"].get("completion_tokens")
                    ch = chunk.get("choices") or []
                    if ch and ch[0].get("text"):
                        n_chunks += 1
                        if t_first is None:
                            t_first = time.perf_counter()
                status = "ok"
    except asyncio.CancelledError:
        status = "cut"
        raise
    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
        status = "timeout" if isinstance(e, asyncio.TimeoutError) else "error"; err = str(e)[:200]
    finally:
        t_end = time.perf_counter()
        n = n_out if n_out is not None else n_chunks
        ok = status == "ok" and t_first is not None
        rows.append({
            "rate": spec["rate"], "i": spec["i"], "tier": spec["tier"],
            "prompt_tokens": len(spec["ids"]), "out_req": spec["out"], "out_got": n,
            "t_sched": f"{clk.wall(spec['t_sched']):.4f}", "t_send": f"{clk.wall(t_send):.4f}",
            "t_first": f"{clk.wall(t_first):.4f}" if t_first else "",
            "t_end": f"{clk.wall(t_end):.4f}",
            "lag_ms": f"{(t_send - spec['t_sched']) * 1000:.1f}",
            # TTFT from the SCHEDULED time: what a user who arrived then would see
            "ttft_ms": f"{(t_first - spec['t_sched']) * 1000:.1f}" if ok else "",
            # TPOT: decode time per token after the first one
            "tpot_ms": f"{(t_end - t_first) / (n - 1) * 1000:.2f}" if ok and n > 1 else "",
            "status": status, "err": err})


def make_specs(rate, dur, t0, rng, trace, tier_w, max_prompt, max_out, vocab_lo, vocab_hi):
    """Pre-draw the whole arrival schedule for one rate (open loop)."""
    specs, t, i, clipped = [], rng.expovariate(rate), 0, 0
    while t < dur:
        p, o = rng.choice(trace)
        clipped += (p > max_prompt) or (o > max_out)
        p, o = min(p, max_prompt), min(o, max_out)
        specs.append({"rate": rate, "i": i, "t_sched": t0 + t,
                      "tier": rng.choices(TIERS, weights=tier_w)[0],
                      "ids": [rng.randrange(vocab_lo, vocab_hi) for _ in range(p)], "out": o})
        t += rng.expovariate(rate); i += 1
    return specs, clipped


async def main_async(a):
    rng = random.Random(a.seed)
    trace = load_trace(os.path.expanduser(a.trace))
    url = f"http://{a.url}/v1/completions"
    os.makedirs(a.out_dir, exist_ok=True)
    clk = Clock(); phases = []
    rows = RowSink(os.path.join(a.out_dir, "e1_requests.csv"), REQ_COLS)   # streamed to disk
    phases_path = os.path.join(a.out_dir, "e1_phases.csv")
    print(f"trace: {len(trace)} requests; median prompt {statistics.median(p for p, _ in trace):.0f} tok, "
          f"median output {statistics.median(o for _, o in trace):.0f} tok; tier mix {a.tier_mix}", flush=True)

    # Unlimited connections: in an open loop the client must never be the queue.
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=0)) as session:
        # Warm-up, discarded: first requests after a (re)start pay one-off costs.
        warm, _ = make_specs(1.0, 5.0, time.perf_counter(), rng, trace, a.tier_mix,
                             a.max_prompt, a.max_out, a.vocab_lo, a.vocab_hi)
        await asyncio.gather(*(one_request(session, url, a.model, s, clk, [], a.timeout_s) for s in warm[:5]),
                             return_exceptions=True)

        def mark(name, m0, m1):
            phases.append((name, clk.wall(m0), clk.wall(m1)))
            write_phases(phases_path, phases)          # on disk after every phase
            print(f"  phase {name:12s} {m1 - m0:6.1f} s", flush=True)

        m = time.perf_counter(); await asyncio.sleep(a.idle_s); mark("idle_pre", m, time.perf_counter())

        for rate in a.rates:
            # Draw the schedule FIRST (at 16 req/s that is ~1M random token ids, ~1 s),
            # then anchor it 1 s in the future, so building it cannot make arrivals late.
            specs, clipped = make_specs(rate, a.dur_s, 0.0, rng, trace, a.tier_mix,
                                        a.max_prompt, a.max_out, a.vocab_lo, a.vocab_hi)
            t0 = time.perf_counter() + 1.0
            for s in specs:
                s["t_sched"] += t0
            tasks = [asyncio.create_task(one_request(session, url, a.model, s, clk, rows, a.timeout_s))
                     for s in specs]
            await asyncio.sleep(max(0.0, t0 + a.dur_s - time.perf_counter()))
            mark(f"rate_{rate:g}", t0, t0 + a.dur_s)
            m = time.perf_counter()
            done, pending = await asyncio.wait(tasks, timeout=a.tail_s)   # let in-flight finish
            for p in pending:
                p.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            mark(f"tail_{rate:g}", m, time.perf_counter())
            print(f"    {len(specs)} requests, {clipped} clipped, {len(pending)} cut at tail limit", flush=True)
            m = time.perf_counter(); await asyncio.sleep(a.gap_s); mark(f"gap_{rate:g}", m, time.perf_counter())
            if len(pending) > 0.5 * max(1, len(specs)):
                print(f"    >50% cut at {rate:g} req/s: saturated; skipping higher rates", flush=True)
                break

        m = time.perf_counter(); await asyncio.sleep(a.idle_s); mark("idle_post", m, time.perf_counter())

    rows.close()                                       # both files are already complete on disk

    # Summary: failures count as +inf in the percentiles, so a p99 of "inf"
    # means more than 1 % of that tier did not complete -- not a good number.
    print(f"\n{'rate':>5s} {'tier':9s} {'n':>4s} {'ok%':>5s} {'TTFT p50':>9s} {'TTFT p99':>9s} "
          f"{'TPOT p50':>9s} {'TPOT p99':>9s} {'lag max':>8s}   (ms)")
    for rate in a.rates:
        for tier in TIERS + ("all",):
            rs = [r for r in rows if r["rate"] == rate and (tier == "all" or r["tier"] == tier)]
            if not rs:
                continue
            ttft = [float(r["ttft_ms"]) if r["ttft_ms"] else math.inf for r in rs]
            # A 1-token answer has no TPOT (nothing after the first token): leave it
            # out. A failed request has none either, but must count as +inf.
            tpot = [float(r["tpot_ms"]) if r["tpot_ms"] else math.inf
                    for r in rs if not (r["status"] == "ok" and r["out_got"] <= 1)]
            ok = sum(r["status"] == "ok" for r in rs)
            print(f"{rate:5g} {tier:9s} {len(rs):4d} {100 * ok / len(rs):5.1f} {pct(ttft, 50):9.1f} "
                  f"{pct(ttft, 99):9.1f} {pct(tpot, 50):9.2f} {pct(tpot, 99):9.2f} "
                  f"{max(float(r['lag_ms']) for r in rs):8.1f}")
    print(f"wrote {a.out_dir}/e1_phases.csv and e1_requests.csv")


def main():
    raise_fd_limit()
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True, help="host:port to load (the proxy, for disaggregated)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--trace", required=True, help="AzureLLMInferenceTrace_conv.csv")
    ap.add_argument("--rates", type=float, nargs="+", default=[0.5, 1, 2, 4, 8, 16], help="req/s, ascending")
    ap.add_argument("--dur-s", type=float, default=60.0, help="arrival window per rate")
    ap.add_argument("--tail-s", type=float, default=60.0, help="max wait for in-flight after the window")
    ap.add_argument("--gap-s", type=float, default=15.0, help="idle between rates (power settles)")
    ap.add_argument("--idle-s", type=float, default=30.0, help="parked baseline before/after")
    ap.add_argument("--tier-mix", type=float, nargs=3, default=[1, 1, 1], help="premium standard flex weights")
    ap.add_argument("--max-prompt", type=int, default=4096)
    ap.add_argument("--max-out", type=int, default=512)
    ap.add_argument("--vocab-lo", type=int, default=100)     # random ids well clear of Qwen's
    ap.add_argument("--vocab-hi", type=int, default=20000)   # special tokens (151643+)
    ap.add_argument("--timeout-s", type=float, default=300.0)
    ap.add_argument("--seed", type=int, default=650)
    ap.add_argument("--out-dir", default=".")
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
