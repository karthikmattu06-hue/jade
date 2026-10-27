#!/usr/bin/env python3
"""
jade_e3_rows.py -- turn an E3 run into datasheet rows (schema v0.3 rev 2 definitions).

INPUT (one log dir): e3_power.csv (sampler), e3_events.csv and e3_requests.csv (jade_e3.py)
OUTPUT: e3_rows.csv (one row per event x pool), e3_rows.json (schema-shaped entries), and a table.

DEFINITIONS (Jade/schema/README.md; repeated here so this file stands alone)
  P        = that pool's GPU power. Instantaneous NVML readings are noisy (decode
             switches between ~60 and ~200 W as requests come and go), so every
             time-based test below uses a CENTRED 0.5 s mean (HALF_W = 0.25 s either side), written
             P1. Centred, not trailing: a trailing mean lags a real drop by ~0.5 s and
             would make every speed look slower than it is (caught on synthetic data).
  P_base   = mean P over the last 30 s of the baseline.
  P_hold   = mean P over the second half of the hold.
  depth    = (P_base - P_hold) / nominal_w
  speed    = t90 - t_cmd; t90 = first time P1 has covered 90 % of the drop P_base -> P_hold.
             "n/a" if the drop is under 5 W (nothing to time).
  duration = time from t_cmd until the first 10 s window in which a SERVED tier's p99
             TTFT exceeds its SLO; if that never happens, the hold length with bound "lower".
             "n/a" for a drain (nothing is served).
  cost     = per tier, p99 TTFT and TPOT during the hold minus during the baseline;
             served_frac = completed / offered during the hold. Requests refused by
             design (HTTP 429) count only in served_frac, never in latency; failed or
             cut requests count as +infinity.
  recovery = t_rec - t_rel; t_rec = first time P1 is within 10 % of P_base AND the
             premium p99 TTFT of the next 10 s (if >= 3 premium requests) is within
             10 % of baseline (or below it).
  rebound  = (max P1 in the after window - P_base) / nominal_w, floored at 0;
  rebound_s= time P1 spends above P_base + 0.05 x nominal_w in the after window.
  recovery90_s (DIAGNOSTIC, not a schema field) = t - t_rel at the first P1 that has won
             back 90 % of the drop. Needed because the schema's recovery test ("within
             10 % of P_base") is met instantly by any lever shallower than 10 %; proposed
             for schema rev 3, to be reviewed with the cloud side first.

Usage:  python jade_e3_rows.py --dir logs_e3_<job> [--nominal-w 250]
"""
import argparse, bisect, csv, json, math, os

SLO_TTFT_MS = {"premium": 2000, "standard": 6000}     # schema v0.3 tiers, status "assumed"
LEVER = {"shed_reject": ("admission_by_tier", {"shed": ["flex"], "mode": "reject"}),
         "shed_defer":  ("admission_by_tier", {"shed": ["flex"], "mode": "defer"}),
         "drain":       ("drain_to_parked", {"shed": ["premium", "standard", "flex"], "mode": "reject"})}
ROLE = {0: "prefill", 1: "decode"}
HALF_W = 0.25   # smoothing half-width (s): 0.5 s centred window. Wider = less noise but blurs fast drops
                # (a 1 s window read a 0.45 s synthetic drop as 0.74 s; 0.5 s reads it within ~0.15 s)


def pct(xs, q):
    if not xs:
        return float("nan")
    xs = sorted(xs); k = min(len(xs) - 1, max(0, math.ceil(q / 100 * len(xs)) - 1))
    return xs[k]


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


class Series:
    """One GPU's power samples with a 1 s rolling mean, searchable by time."""
    def __init__(self, ts, ws):
        self.t, self.w = ts, ws
        # centred 1 s mean via prefix sums: average of samples within +-0.5 s of each sample
        pre = [0.0]
        for x in ws:
            pre.append(pre[-1] + x)
        self.p1 = []
        for i, t in enumerate(ts):
            lo, hi = bisect.bisect_left(ts, t - HALF_W), bisect.bisect_right(ts, t + HALF_W)
            self.p1.append((pre[hi] - pre[lo]) / (hi - lo))

    def idx(self, t):
        return bisect.bisect_left(self.t, t)

    def window(self, a, b, smooth=False):
        src = self.p1 if smooth else self.w
        return src[self.idx(a):self.idx(b)], self.t[self.idx(a):self.idx(b)]


def lat(reqs, tier, a, b, field):
    """p99 of field for tier with t_sched in [a, b); 429s excluded; failures = +inf."""
    xs = []
    for r in reqs:
        if r["tier"] != tier or not (a <= r["ts"] < b) or r["status"] == "http429":
            continue
        xs.append(float(r[field]) if r[field] else math.inf)
    return pct(xs, 99), len(xs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--nominal-w", type=float, default=250.0)
    ap.add_argument("--hardware-ref", default="sporc_a100_pcie_disagg")
    a = ap.parse_args(); NOM = a.nominal_w
    rd = lambda f: list(csv.DictReader(open(os.path.join(a.dir, f))))
    power = {}
    for r in rd("e3_power.csv"):
        power.setdefault(int(r["gpu"]), ([], []))
        power[int(r["gpu"])][0].append(float(r["t_wall"])); power[int(r["gpu"])][1].append(float(r["power_w"]))
    series = {g: Series(*v) for g, v in power.items()}
    reqs = rd("e3_requests.csv")
    for r in reqs:
        r["ts"] = float(r["t_sched"])
    out_rows, out_json = [], []

    for ev in rd("e3_events.csv"):
        t0, tc, tr, te = (float(ev[k]) for k in ("t_start", "t_cmd", "t_rel", "t_end"))
        er = [r for r in reqs if r["event"] == ev["event"]]
        kind, rate = ev["kind"], float(ev["rate"])
        hold = tr - tc
        # ---- request-side rows (shared by both pools of this event) ----------
        cost, base_p99 = {}, {}
        for tier in ("premium", "standard", "flex"):
            b_ttft, _ = lat(er, tier, t0, tc, "ttft_ms"); h_ttft, n_h = lat(er, tier, tc, tr, "ttft_ms")
            b_tpot, _ = lat(er, tier, t0, tc, "tpot_ms"); h_tpot, _ = lat(er, tier, tc, tr, "tpot_ms")
            offered = [r for r in er if r["tier"] == tier and tc <= r["ts"] < tr]
            served = sum(r["status"] == "ok" for r in offered)
            base_p99[tier] = b_ttft
            cost[tier] = {"d_ttft_p99_ms": h_ttft - b_ttft if n_h else None,
                          "d_tpot_p99_ms": h_tpot - b_tpot if n_h else None,
                          "served_frac": served / len(offered) if offered else None}
        # duration: 10 s windows through the hold; first window where a served tier breaks its SLO
        duration, bound = None, None
        if kind != "drain":
            duration, bound, w = hold, "lower", tc
            while w + 10 <= tr + 1e-6:
                broke = any(lat(er, t, w, w + 10, "ttft_ms")[0] > SLO_TTFT_MS[t]
                            for t in SLO_TTFT_MS if lat(er, t, w, w + 10, "ttft_ms")[1] > 0)
                if broke:
                    duration, bound = w - tc, None; break
                w += 10
        d_after = lat(er, "premium", tr, te, "ttft_ms")[0] - base_p99["premium"]

        for g, s in sorted(series.items()):
            P_base = mean(s.window(max(t0, tc - 30), tc)[0])   # never reach into the previous event
            hw, _ = s.window(tc + hold / 2, tr); P_hold = mean(hw)
            depth = (P_base - P_hold) / NOM
            # speed: first t where the 1 s mean has covered 90 % of the drop
            speed = None
            if P_base - P_hold >= 5:
                target = P_base - 0.9 * (P_base - P_hold)
                p1, ts = s.window(tc, tr, smooth=True)
                hit = next((t for p, t in zip(p1, ts) if p <= target), None)
                speed = hit - tc if hit is not None else None
            # release
            p1, ts = s.window(tr, te, smooth=True)
            rec = None
            for p, t in zip(p1, ts):
                if abs(p - P_base) <= 0.1 * P_base:
                    l99, n = lat(er, "premium", t, t + 10, "ttft_ms")
                    if n < 3 or l99 <= 1.1 * base_p99["premium"]:
                        rec = t - tr; break
            rebound = max(0.0, (max(p1) - P_base) / NOM) if p1 else None
            rec90 = None
            if P_base - P_hold >= 5:
                back = P_hold + 0.9 * (P_base - P_hold)
                rec90 = next((t - tr for p, t in zip(p1, ts) if p >= back), None)
            dt = mean([b - a_ for a_, b in zip(ts, ts[1:])]) if len(ts) > 1 else 0.05
            reb_s = sum(dt for p in p1 if p > P_base + 0.05 * NOM)
            row = {"event": ev["event"], "rate": rate, "kind": kind, "pool": ROLE.get(g, g),
                   "P_base_w": round(P_base, 1), "P_hold_w": round(P_hold, 1),
                   "busy_frac": round(P_base / NOM, 3), "depth_frac": round(depth, 3),
                   "speed_s": None if speed is None else round(speed, 2),
                   "duration_s": None if duration is None else round(duration, 1), "duration_bound": bound,
                   "d_ttft_p99_prem_ms": cost["premium"]["d_ttft_p99_ms"],
                   "d_ttft_p99_std_ms": cost["standard"]["d_ttft_p99_ms"],
                   "flex_served_frac": cost["flex"]["served_frac"],
                   "recovery_s": None if rec is None else round(rec, 2),
                   "recovery90_s": None if rec90 is None else round(rec90, 2),
                   "rebound_frac": None if rebound is None else round(rebound, 3),
                   "rebound_s": round(reb_s, 1), "d_ttft_p99_after_ms": d_after}
            out_rows.append(row)
            name, setting = LEVER[kind]
            st = lambda v: {"value": v, "status": "measured" if v is not None else "n/a"}
            out_json.append({
                "name": name, "pool": ROLE.get(g, g), "authority": "operator", "setting": setting,
                "at_load": {"rate_rps": rate, "pool_busy_frac": row["busy_frac"],
                            "tier_mix": {"premium": 0.333, "standard": 0.333, "flex": 0.334}, "status": "measured"},
                "source_hardware": {"ref": a.hardware_ref, "event": ev["event"]},
                "depth_frac": st(row["depth_frac"]), "speed_s": st(row["speed_s"]),
                "duration_s": dict(st(row["duration_s"]), bound=bound),
                "cost": {t: {k: st(v) for k, v in c.items()} for t, c in cost.items()},
                "release": {"recovery_s": st(row["recovery_s"]), "rebound_frac": st(row["rebound_frac"]),
                            "rebound_s": st(row["rebound_s"]), "d_ttft_p99_after_ms": st(d_after)}})

    with open(os.path.join(a.dir, "e3_rows.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0])); w.writeheader(); w.writerows(out_rows)
    with open(os.path.join(a.dir, "e3_rows.json"), "w") as f:
        json.dump(out_json, f, indent=1, default=lambda x: None if isinstance(x, float) and math.isnan(x) else x)

    fmt = lambda v, n=1: "-" if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))) else f"{v:.{n}f}"
    print(f"{'ev':5s} {'kind':11s} {'rps':>3s} {'pool':7s} {'Pbase':>6s} {'Phold':>6s} {'depth':>6s} {'speed':>6s} "
          f"{'dur':>7s} {'dTTFTprem':>9s} {'flexSrv':>7s} {'recov':>6s} {'rec90':>6s} {'rebnd':>6s}")
    for r in out_rows:
        dur = fmt(r["duration_s"], 0) + ("+" if r["duration_bound"] == "lower" else "")
        print(f"{r['event']:5s} {r['kind']:11s} {r['rate']:3g} {r['pool']:7s} {r['P_base_w']:6.1f} {r['P_hold_w']:6.1f} "
              f"{r['depth_frac']:6.3f} {fmt(r['speed_s'], 2):>6s} {dur:>7s} {fmt(r['d_ttft_p99_prem_ms'], 0):>9s} "
              f"{fmt(r['flex_served_frac'], 2):>7s} {fmt(r['recovery_s'], 1):>6s} {fmt(r['recovery90_s'], 1):>6s} {fmt(r['rebound_frac'], 3):>6s}")
    print(f"wrote {a.dir}/e3_rows.csv and e3_rows.json   ('+' on duration = lower bound: no SLO broke in the hold)")


if __name__ == "__main__":
    main()
