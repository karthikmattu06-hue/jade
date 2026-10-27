#!/usr/bin/env python3
"""
jade_smoke_plot.py -- per-phase power table + time-series plot for the v6 smoke test.

Reads, from one log directory:
    power.csv     (jade_power_sampler.py)
    phases.csv    (disagg_smoke_client.py v2)
    requests.csv  (disagg_smoke_client.py v2)
Prints a table of every phase x GPU and, if matplotlib is importable, saves
power_trace.png. Standard library only for the table, so it also runs on the
compute node; the plot is optional (run it on your Mac if the env lacks matplotlib).

What to look for:
  * idle_pre / idle_post  -> parked watts per GPU (E0.1 saw ~39-43 W)
  * disagg                -> decode GPU busy (~190-200 W); prefill GPU only blips,
                             because a 180-token prefill takes ~10 ms per 1.3 s request
  * mono                  -> ONLY GPU 1 busy. If GPU 0 rises here, roles are swapped.
  * energy_J uses the hardware counter (last - first in the window): trust it over
    mean_W x seconds.

Usage:  python jade_smoke_plot.py --dir logs_<job>              # smoke test files
        python jade_smoke_plot.py --dir logs_<job> --prefix e1_ # E1: e1_power.csv, e1_phases.csv, ...
"""
import argparse, csv, os, statistics
from collections import defaultdict


def load(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--prefix", default="", help="file prefix, e.g. e1_")
    a = ap.parse_args()
    power = load(os.path.join(a.dir, a.prefix + "power.csv"))
    phases = [(p["phase"], float(p["t_start"]), float(p["t_end"])) for p in load(os.path.join(a.dir, a.prefix + "phases.csv"))]
    reqs = load(os.path.join(a.dir, a.prefix + "requests.csv"))

    by_gpu = defaultdict(list)                 # gpu -> rows sorted by time
    for r in power:
        by_gpu[(int(r["gpu"]), r["role"])].append(r)

    print(f"{'phase':10s} {'gpu':>3s} {'role':8s} {'sec':>6s} {'mean_W':>7s} {'inst_W':>7s} "
          f"{'energy_J':>9s} {'pcap%':>6s} {'SM_MHz':>7s} {'temp_C':>6s}")
    for name, t0, t1 in phases:
        for (g, role), rows in sorted(by_gpu.items()):
            win = [r for r in rows if t0 <= float(r["t_wall"]) <= t1]
            if len(win) < 2:
                continue
            pw = [float(r["power_w"]) for r in win]
            pi = [float(r["power_inst_w"]) for r in win if r["power_inst_w"]]
            ej = float(win[-1]["energy_j"]) - float(win[0]["energy_j"])
            print(f"{name:10s} {g:3d} {role:8s} {t1-t0:6.1f} {statistics.mean(pw):7.1f} "
                  f"{(statistics.mean(pi) if pi else float('nan')):7.1f} {ej:9.1f} "
                  f"{100*sum(int(r['pcap']) for r in win)/len(win):6.1f} "
                  f"{statistics.mean(int(r['sm_mhz']) for r in win):7.0f} "
                  f"{statistics.mean(int(r['temp_c']) for r in win):6.1f}")

    try:
        import matplotlib
        matplotlib.use("Agg")                  # no display on a compute node
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available -- table only; copy the dir to your Mac to plot")
        return

    t_ref = phases[0][1] - 5
    t_end = phases[-1][2] + 5
    fig, (ax_w, ax_c) = plt.subplots(2, 1, figsize=(12, 6.5), sharex=True,
                                     gridspec_kw={"height_ratios": [3, 1.3]})
    shade = {"disagg": "#cfe3f7", "mono": "#f7e3cf"}
    for name, t0, t1 in phases:                # shade the busy phases, label all
        label = name if not name.startswith(("tail_", "gap_")) else ""
        if name in shade or name.startswith("rate_"):
            for ax in (ax_w, ax_c):
                ax.axvspan(t0 - t_ref, t1 - t_ref, color=shade.get(name, "#d9ead3"), alpha=0.6, lw=0)
        ax_w.text((t0 + t1) / 2 - t_ref, 1.01, label, transform=ax_w.get_xaxis_transform(),
                  ha="center", va="bottom", fontsize=9)
    for (g, role), rows in sorted(by_gpu.items()):
        rows = [r for r in rows if t_ref <= float(r["t_wall"]) <= t_end]
        t = [float(r["t_wall"]) - t_ref for r in rows]
        col = "C0" if g == 0 else "C1"
        ax_w.plot(t, [float(r["power_w"]) for r in rows], color=col, lw=1.2,
                  label=f"GPU {g} {role} (NVML avg)")
        if rows and rows[0]["power_inst_w"]:
            ax_w.plot(t, [float(r["power_inst_w"]) if r["power_inst_w"] else float("nan") for r in rows],
                      color=col, lw=0.6, alpha=0.5, label=f"GPU {g} {role} (instant)")
        ax_c.plot(t, [int(r["sm_mhz"]) for r in rows], color=col, lw=1)
    for r in reqs:                             # one tick per request send
        ax_w.axvline(float(r["t_send"]) - t_ref, ymin=0, ymax=0.04,
                     color="C0" if r.get("path", "disagg") == "disagg" else "C1", lw=0.8)
    ax_w.set_ylabel("power (W)"); ax_w.legend(loc="upper right", fontsize=8); ax_w.set_ylim(bottom=0)
    ax_c.set_ylabel("SM clock (MHz)"); ax_c.set_xlabel("time (s)")
    fig.tight_layout()
    out = os.path.join(a.dir, a.prefix + "power_trace.png")
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
