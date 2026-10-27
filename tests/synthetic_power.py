#!/usr/bin/env python3
"""
synthetic_power.py -- write an e3_power.csv with KNOWN answers, to test jade_e3_rows.py.

For every event in e3_events.csv it draws two GPUs at 20 Hz with small noise:
  baseline: prefill 60 W, decode 200 W
  drain:    both fall to 40 W over 0.5 s       -> decode depth (200-40)/250 = 0.64
  shed:     prefill 55 W, decode 180 W          -> decode depth 0.08
  defer:    +30 W for 1.5 s after release       -> rebound 30/250 = 0.12
The test then checks jade_e3_rows.py recovers these numbers.

Usage:  python synthetic_power.py <log dir containing e3_events.csv>
"""
import csv, os, random, sys

d = sys.argv[1]
ev = list(csv.DictReader(open(os.path.join(d, "e3_events.csv"))))
rng = random.Random(1)
DROP = {"drain": {0: 40, 1: 40}, "shed_reject": {0: 55, 1: 180}, "shed_defer": {0: 55, 1: 180}}


def power(g, t):
    for e in ev:
        t0, tc, tr, te = (float(e[k]) for k in ("t_start", "t_cmd", "t_rel", "t_end"))
        if t0 <= t < te + 0.5:
            base = {0: 60, 1: 200}[g]
            if tc <= t < tr:
                return base + (DROP[e["kind"]][g] - base) * min(1, (t - tc) / 0.5)
            if e["kind"] == "shed_defer" and tr <= t < tr + 1.5:
                return base + 30
            return base
    return 40


t, t_end = float(ev[0]["t_start"]) - 5, float(ev[-1]["t_end"]) + 5
with open(os.path.join(d, "e3_power.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["t_wall", "gpu", "role", "bus_id", "power_w", "power_inst_w", "energy_j",
                "sm_mhz", "util_pct", "temp_c", "throttle", "pcap"])
    while t < t_end:
        for g in (0, 1):
            w.writerow([f"{t:.3f}", g, "prefill" if g == 0 else "decode", "mock",
                        f"{power(g, t) + rng.gauss(0, 2):.1f}", "", 0, 1410, 0, 40, 0, 0])
        t += 0.05
