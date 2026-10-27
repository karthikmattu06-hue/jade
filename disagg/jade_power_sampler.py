#!/usr/bin/env python3
"""
jade_power_sampler.py -- per-GPU power / clock / throttle logger for Jade.

WHAT IT DOES
  Polls NVML (the NVIDIA management library that nvidia-smi is built on) at a
  fixed rate -- 20 Hz by default, the same rate E0/E0.1 used -- and writes ONE
  CSV ROW PER GPU PER TICK. It runs as its own process next to the vLLM servers,
  starts before them, and stops when it gets SIGTERM (the job script sends it).
  Reading NVML is unprivileged; nothing here needs root. Writing a power cap
  does, which is why the cap experiments go to GCP.

WHY A SEPARATE FILE
  E1 and E3 need exactly this logger. Keeping it standalone means the smoke
  test, E1 and E3 all produce the same CSV, so one plotting / analysis script
  reads all of them.

CLOCK
  Every row carries t_wall = time.time() (seconds since 1970, wall clock). The
  load client stamps its requests with the same clock, on the same node, so
  power samples and request events line up without any synchronisation step.

COLUMNS
  t_wall        wall-clock time of the sample (s)
  gpu           NVML index (0, 1, ...). With CUDA_DEVICE_ORDER=PCI_BUS_ID set in
                the job, this equals the CUDA index each vLLM server was pinned to.
  role          label you pass in (prefill / decode) -- for plots only
  bus_id        PCI bus id; the ground truth for "which physical card"
  power_w       nvmlDeviceGetPowerUsage. CAUTION: on Ampere (A100) and newer this
                is a ~1 s RUNNING AVERAGE, not an instantaneous reading. Fine
                for steady state; it smears transients over ~1 s.
  power_inst_w  NVML_FI_DEV_POWER_INSTANT field, if the driver exposes it (newer
                drivers do). Instantaneous. This is the column to use when you
                measure how FAST power falls during a drain event (E3 "speed").
                Empty if unsupported -- the sampler prints which at startup.
  energy_j      hardware energy counter, joules since the sampler started. The
                most trustworthy energy number: integrate nothing, just take
                (last - first) over a window. ~100 ms granularity.
  sm_mhz        current SM (compute) clock. Drops when a cap or heat bites.
  util_pct      % of time a kernel was running over the last sample period
  temp_c        GPU temperature. Parked power drifts ~0.4 W/degC (E0.1).
  throttle      raw clock-event (throttle-reason) bitmask from the driver
  pcap          1 if the SW_POWER_CAP bit is set -- "the power limit is actively
                pulling clocks down right now". E0.1: mean power vs limit is NOT
                headroom; this flag is.

USAGE
  python jade_power_sampler.py --gpus 0 1 --roles prefill decode --out power.csv
"""
import argparse
import csv
import signal
import sys
import time

import pynvml  # ships with vLLM's dependencies (package name: nvidia-ml-py)

# NVIDIA renamed the throttle API ("ClocksThrottleReasons" -> "ClocksEventReasons").
# Bind whichever this pynvml has, exactly as energy_probe.py does, so a pynvml
# upgrade cannot kill a run.
_THROTTLE_FN = (getattr(pynvml, "nvmlDeviceGetCurrentClocksThrottleReasons", None)
                or getattr(pynvml, "nvmlDeviceGetCurrentClocksEventReasons", None))
BIT_POWER_CAP = (getattr(pynvml, "nvmlClocksEventReasonSwPowerCap", None)
                 or getattr(pynvml, "nvmlClocksThrottleReasonSwPowerCap", None)
                 or 0x4)
# Field id 186 in nvml.h; use the named constant when the binding has it.
FI_POWER_INSTANT = getattr(pynvml, "NVML_FI_DEV_POWER_INSTANT", 186)

# NVML field values come back as a C union; valueType says which member is live.
_UNION_MEMBER = {0: "dVal", 1: "uiVal", 2: "ulVal", 3: "ullVal", 4: "sllVal", 5: "siVal"}


def read_instant_w(h):
    """Instantaneous power in watts, or None if the driver does not expose it."""
    try:
        fv = pynvml.nvmlDeviceGetFieldValues(h, [FI_POWER_INSTANT])[0]
        if fv.nvmlReturn != pynvml.NVML_SUCCESS:
            return None
        member = _UNION_MEMBER.get(int(fv.valueType), "uiVal")
        return float(getattr(fv.value, member)) / 1000.0   # milliwatts -> watts
    except (pynvml.NVMLError, AttributeError, TypeError, IndexError):
        return None


def read_throttle(h) -> int:
    if _THROTTLE_FN is None:
        return -1
    try:
        return int(_THROTTLE_FN(h))
    except pynvml.NVMLError:
        return -1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", type=int, nargs="+", default=[0, 1], help="NVML indices to log")
    ap.add_argument("--roles", nargs="+", default=None, help="one label per GPU, e.g. prefill decode")
    ap.add_argument("--hz", type=float, default=20.0)
    ap.add_argument("--out", required=True, help="CSV path")
    ap.add_argument("--max-s", type=float, default=3600.0,
                    help="safety stop: never outlive the job even if SIGTERM is missed")
    a = ap.parse_args()
    roles = a.roles or [f"gpu{g}" for g in a.gpus]
    if len(roles) != len(a.gpus):
        sys.exit("--roles must have one entry per --gpus entry")

    # SIGTERM (sent by the job script, or by Slurm at time-out) just sets a flag;
    # the loop then exits cleanly and the CSV is flushed and closed.
    stop = {"now": False}
    def _stop(*_):
        stop["now"] = True
    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    pynvml.nvmlInit()
    handles = [pynvml.nvmlDeviceGetHandleByIndex(g) for g in a.gpus]
    bus = []
    for g, h in zip(a.gpus, handles):
        b = pynvml.nvmlDeviceGetPciInfo(h).busId
        bus.append(b.decode() if isinstance(b, bytes) else b)
    e0 = [pynvml.nvmlDeviceGetTotalEnergyConsumption(h) for h in handles]   # millijoules
    has_inst = [read_instant_w(h) is not None for h in handles]

    for g, r, b, hi in zip(a.gpus, roles, bus, has_inst):
        lim = pynvml.nvmlDeviceGetEnforcedPowerLimit(pynvml.nvmlDeviceGetHandleByIndex(g)) / 1000
        print(f"[sampler] gpu {g} role={r} bus={b} limit={lim:.0f} W "
              f"instant-power field: {'yes' if hi else 'NO (power_inst_w will be empty)'}", flush=True)
    print(f"[sampler] {a.hz:g} Hz -> {a.out}", flush=True)

    period = 1.0 / a.hz
    t_start = time.perf_counter()
    next_tick = t_start
    n_ticks = 0
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_wall", "gpu", "role", "bus_id", "power_w", "power_inst_w", "energy_j",
                    "sm_mhz", "util_pct", "temp_c", "throttle", "pcap"])
        while not stop["now"] and time.perf_counter() - t_start < a.max_s:
            t_wall = time.time()
            for i, (g, h) in enumerate(zip(a.gpus, handles)):
                try:
                    p = pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0
                    pi = read_instant_w(h) if has_inst[i] else None
                    ej = (pynvml.nvmlDeviceGetTotalEnergyConsumption(h) - e0[i]) / 1000.0
                    sm = pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM)
                    ut = pynvml.nvmlDeviceGetUtilizationRates(h).gpu
                    tc = pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
                except pynvml.NVMLError as err:      # one bad read must not end the log
                    print(f"[sampler] NVML read failed on gpu {g}: {err}", flush=True)
                    continue
                th = read_throttle(h)
                w.writerow([f"{t_wall:.4f}", g, roles[i], bus[i], f"{p:.2f}",
                            "" if pi is None else f"{pi:.2f}", f"{ej:.3f}",
                            sm, ut, tc, th, int(th >= 0 and bool(th & BIT_POWER_CAP))])
            n_ticks += 1
            if n_ticks % max(1, int(a.hz)) == 0:     # flush ~once a second: a killed
                f.flush()                            # job still leaves a usable CSV
            # Fixed-rate schedule: aim for t_start + k*period. If we fell behind
            # (a slow NVML call), skip ahead instead of firing a burst of catch-up ticks.
            next_tick += period
            delay = next_tick - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            else:
                next_tick = time.perf_counter()
    pynvml.nvmlShutdown()
    print(f"[sampler] stopped after {n_ticks} ticks "
          f"({n_ticks / max(1e-9, time.perf_counter() - t_start):.1f} Hz achieved)", flush=True)


if __name__ == "__main__":
    main()
