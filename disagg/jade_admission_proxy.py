#!/usr/bin/env python3
"""
jade_admission_proxy.py -- tier-aware admission control: Jade's first load lever.

WHERE IT SITS
    load generator --> THIS PROXY (:8300) --> vLLM toy proxy (:8192) --> prefill GPU / decode GPU
It reads the X-Jade-Tier header (premium / standard / flex) that jade_loadgen.py
sends, and decides per request whether to let it through.

THE LEVER
  When a budget event is active, requests of the "shed" tiers are not served:
    mode "reject" -- answered at once with HTTP 429. The work is LOST. Power drops
                     because the GPUs get less to do; nothing comes back later.
    mode "defer"  -- held here until the event ends (or --max-hold-s passes, then
                     429). The work is POSTPONED. When the event ends, every held
                     request is released at once -- that burst is the REBOUND the
                     datasheet's "release" row measures (the Phoenix utility terms
                     ask for "no snap-back"). Reject vs defer is a real policy
                     choice for an operator, so E3 measures both.
  RAMPED RELEASE (added 11 Nov 2026): end a defer event with
                    {"shed": [], "ramp_s": 30}
                   and the held requests are let go one by one, spaced evenly
                   over 30 s, oldest first, instead of all at once. E3 on SPORC
                   showed the all-at-once release spikes the prefill GPU to
                   ~250 W; a ramp is the obvious answer, and this measures it.
                   New requests arriving during the ramp pass straight through
                   (the event is over); only the backlog is metered.
  Shedding every tier ("shed": ["premium","standard","flex"]) is a full DRAIN:
  new work stops, in-flight requests finish, both pools fall to parked power.

CONTROL (the event driver or you, with curl)
    curl -s -X POST localhost:8300/control -d '{"shed":["flex"],"mode":"defer","note":"event start"}'
    curl -s -X POST localhost:8300/control -d '{"shed":[],"note":"event end"}'
    curl -s -X POST localhost:8300/control -d '{"shed":[],"ramp_s":30,"note":"event end, ramped"}'
    curl -s localhost:8300/state
  Every command is written to commands.csv with a wall-clock timestamp (same
  clock as the power sampler): "command at t" -> "power settled at t + x" is
  how the SPEED row is measured.

LOGS (in --log-dir)
    commands.csv   : t_wall, shed tiers, mode, in-flight and held counts, note
    admissions.csv : one row per request -- tier, decision, hold time

STREAMING
  Responses are passed through chunk by chunk as they arrive (iter_any), never
  buffered, so time-to-first-token is not inflated by this hop. The hop itself
  costs a local HTTP round trip (well under 1 ms); E2 measures it.
"""
import argparse, asyncio, collections, csv, json, os, time

from aiohttp import ClientSession, ClientTimeout, TCPConnector, web

TIERS = ("premium", "standard", "flex")


class State:
    def __init__(self, log_dir, max_hold_s):
        self.shed, self.mode = set(), "reject"
        # Each held request parks on its own Future in this FIFO queue. Releasing
        # = resolving futures: all at once (instant release) or one per interval
        # (ramped release). A plain asyncio.Event could only do "all at once".
        self.queue = collections.deque()
        self.ramp_task = None
        self.inflight = self.held = 0
        self.max_hold_s = max_hold_s
        os.makedirs(log_dir, exist_ok=True)
        self._cf = open(os.path.join(log_dir, "commands.csv"), "w", newline="")
        self._af = open(os.path.join(log_dir, "admissions.csv"), "w", newline="")
        self.cmd = csv.writer(self._cf); self.adm = csv.writer(self._af)
        self.cmd.writerow(["t_wall", "shed", "mode", "inflight", "held", "note"])
        self.adm.writerow(["t_in", "tier", "decision", "hold_ms"])
        self._n = 0

    def log_cmd(self, note):
        self.cmd.writerow([f"{time.time():.4f}", "|".join(sorted(self.shed)), self.mode,
                           self.inflight, self.held, note]); self._cf.flush()

    def log_adm(self, t_in, tier, decision, hold_s):
        self.adm.writerow([f"{t_in:.4f}", tier, decision, f"{hold_s * 1000:.1f}"])
        self._af.flush()        # every row: a killed proxy must not lose the tail (~30 rows/s at most, cheap)

    def close(self):
        self._cf.close(); self._af.close()

    def snapshot(self):
        return {"shed": sorted(self.shed), "mode": self.mode,
                "inflight": self.inflight, "held": self.held, "t_wall": time.time()}


def release_all(st):
    """Resolve every waiting future: all held requests proceed now."""
    while st.queue:
        fut = st.queue.popleft()
        if not fut.done():
            fut.set_result(True)


async def ramp_release(st, ramp_s):
    """Let the backlog go one request at a time, evenly spaced over ramp_s.
    The spacing is fixed when the ramp starts (backlog size / ramp length)."""
    n = len(st.queue)
    gap = ramp_s / n
    for _ in range(n):
        while st.queue and st.queue[0].done():       # skip requests that already timed out
            st.queue.popleft()
        if not st.queue:
            break
        st.queue.popleft().set_result(True)
        await asyncio.sleep(gap)
    release_all(st)             # anything that slipped in meanwhile (should be none)


async def control(request):
    """POST {"shed": [...], "mode": "reject"|"defer", "note": "..."} -- apply and log."""
    st = request.app["st"]
    try:
        cmd = await request.json()
        shed = set(cmd.get("shed", []))
        mode = cmd.get("mode", st.mode)
        ramp_s = float(cmd.get("ramp_s", 0))
        assert shed <= set(TIERS) and mode in ("reject", "defer") and ramp_s >= 0
    except Exception:
        return web.json_response({"error": f"bad command; tiers {TIERS}, mode reject|defer, ramp_s >= 0"}, status=400)
    st.shed, st.mode = shed, mode
    if not shed:                # event over: let the backlog go
        if st.ramp_task and not st.ramp_task.done():
            st.ramp_task.cancel()                    # a new release replaces an unfinished ramp
        if ramp_s > 0 and st.queue:
            st.ramp_task = asyncio.create_task(ramp_release(st, ramp_s))
        else:
            release_all(st)     # every held request goes at once (the rebound)
    st.log_cmd(f"{cmd.get('note', '')}{f' [ramp {ramp_s:g}s, {len(st.queue)} held]' if not shed and ramp_s else ''}")
    print(f"[admission] {time.strftime('%H:%M:%S')} shed={sorted(shed)} mode={mode} "
          f"inflight={st.inflight} held={st.held}", flush=True)
    return web.json_response(st.snapshot())


async def state(request):
    return web.json_response(request.app["st"].snapshot())


async def forward(request):
    """Every other path (/v1/completions, /v1/chat/completions, ...) goes upstream."""
    st, app = request.app["st"], request.app
    tier = request.headers.get("X-Jade-Tier", "standard")
    t_in = time.time(); body = await request.read(); hold = 0.0

    if tier in st.shed:
        if st.mode == "reject":
            st.log_adm(t_in, tier, "reject", 0.0)
            return web.json_response({"error": "shed by jade admission control"}, status=429)
        st.held += 1; m0 = time.perf_counter()
        fut = asyncio.get_running_loop().create_future()
        st.queue.append(fut)
        try:                                   # wait for release, but not forever
            await asyncio.wait_for(asyncio.shield(fut), timeout=st.max_hold_s)
        except asyncio.TimeoutError:
            fut.cancel()                       # the ramp skips cancelled futures
            st.held -= 1
            st.log_adm(t_in, tier, "defer_timeout", time.perf_counter() - m0)
            return web.json_response({"error": "held past max-hold, dropped"}, status=429)
        st.held -= 1; hold = time.perf_counter() - m0

    st.log_adm(t_in, tier, "deferred" if hold else "pass", hold)
    st.inflight += 1
    try:
        async with app["session"].request(request.method, app["upstream"] + request.path_qs, data=body,
                                       headers={"Content-Type": "application/json"}) as up:
            resp = web.StreamResponse(status=up.status, headers={
                "Content-Type": up.headers.get("Content-Type", "application/json")})
            await resp.prepare(request)
            async for chunk in up.content.iter_any():   # pass bytes on as they arrive
                await resp.write(chunk)
            await resp.write_eof()
            return resp
    except (ConnectionResetError, asyncio.CancelledError):
        raise                                           # client went away; nothing to send
    finally:
        st.inflight -= 1


async def on_startup(app):
    app["session"] = ClientSession(connector=TCPConnector(limit=0),
                                   timeout=ClientTimeout(total=None, sock_read=600))
    app["st"].log_cmd("proxy start")


async def on_cleanup(app):
    await app["session"].close(); app["st"].close()


def main():
    import resource                    # held requests = open sockets; see jade_loadgen.raise_fd_limit
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft < hard:
        resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8300)
    ap.add_argument("--upstream", default="localhost:8192", help="host:port of the vLLM toy proxy")
    ap.add_argument("--log-dir", default=".")
    ap.add_argument("--max-hold-s", type=float, default=600.0, help="defer mode: drop after this long; >= 600 s so the long-hold event (schema v0.3 M3) is not cut")
    a = ap.parse_args()
    app = web.Application(client_max_size=64 * 1024 ** 2)   # long token-id prompts are big JSON
    app["upstream"] = f"http://{a.upstream}"
    app["st"] = State(a.log_dir, a.max_hold_s)
    app.router.add_post("/control", control)
    app.router.add_get("/state", state)
    app.router.add_route("*", "/{tail:.*}", forward)
    app.on_startup.append(on_startup); app.on_cleanup.append(on_cleanup)
    print(f"[admission] :{a.port} -> {app['upstream']}  logs in {a.log_dir}", flush=True)
    web.run_app(app, port=a.port, print=None)


if __name__ == "__main__":
    main()
