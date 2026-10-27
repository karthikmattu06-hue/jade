#!/usr/bin/env python3
"""
lambda_cap_probe.py -- does a power cap / clock lock REALLY take effect on this VM?

WHY THIS EXISTS
  On some clouds `sudo nvidia-smi -pl 200` returns "OK" but the hypervisor or
  firmware ignores it. The only proof is behavioural: put the GPU under load,
  set the cap, and check the measured power actually stays under it (and the
  driver raises its SW_POWER_CAP throttle flag). This script does exactly that,
  so you find out in minutes, before paying for 30 GPU-hours.

WHAT IT DOES (about 3 min per load type)
  1. Starts a synthetic load on one GPU in a child process:
       compute  -- big bf16 matrix multiplies: compute-bound, like PREFILL
       memory   -- large tensor copies: memory-bandwidth-bound, like DECODE
  2. Samples NVML at 20 Hz (power, SM clock, throttle flags) -- same idea as
     Jade/disagg/jade_power_sampler.py, but standalone so this file needs nothing else.
  3. Steps the power cap down a ladder (default limit -> each cap -> back to
     default), holding each step, then locks the SM clock and releases it.
     Every command is logged with a timestamp.
  4. Prints, per step: mean / max watts, % of samples with the cap flag, mean
     clock, and the SPEED: seconds from command until power first falls to
     within 5 % of the new cap. That is the datasheet's "speed" row in miniature.

PASS (printed at the end)
  cap works   -- in every cap step that BINDS (unconstrained power > cap), mean
                 power during the second half of the hold is <= cap + 5 %.
  clock works -- during the lock step, mean SM clock <= locked value + 15 MHz.
  If the load never exceeds a cap (likely for "memory" at high caps), that step
  is reported as "not binding", which is a RESULT (G6: does a cap bind decode?),
  not a failure.

NEEDS: sudo without a password (Lambda's `ubuntu` user has it), torch (Lambda
Stack ships it), nvidia-ml-py (`pip install nvidia-ml-py`).

USAGE
  python3 lambda_cap_probe.py --gpu 0 --load compute --out probe_compute
  python3 lambda_cap_probe.py --gpu 0 --load memory  --out probe_memory
"""
import argparse, csv, os, subprocess, sys, time

import pynvml

LOAD_CODE = {
    # Each child runs until killed. Sizes are chosen to keep an A100 busy.
    "compute": (
        "import torch; a=torch.randn(8192,8192,device='cuda',dtype=torch.bfloat16); b=a.clone()\n"
        "while True:\n"
        "    for _ in range(20): c=a@b\n"
        "    torch.cuda.synchronize()\n"),
    "memory": (
        "import torch; x=torch.empty(int(2.5e9)//2,device='cuda',dtype=torch.bfloat16); y=torch.empty_like(x)\n"
        "while True:\n"
        "    for _ in range(20): y.copy_(x)\n"
        "    torch.cuda.synchronize()\n"),
}
BIT_POWER_CAP = getattr(pynvml, "nvmlClocksEventReasonSwPowerCap", None) or \
                getattr(pynvml, "nvmlClocksThrottleReasonSwPowerCap", None) or 0x4
_THROTTLE = (getattr(pynvml, "nvmlDeviceGetCurrentClocksEventReasons", None)
             or getattr(pynvml, "nvmlDeviceGetCurrentClocksThrottleReasons", None))


def smi(*args):
    """Run sudo nvidia-smi; return (ok, output). Logged by the caller."""
    p = subprocess.run(["sudo", "-n", "nvidia-smi", *args], capture_output=True, text=True)
    return p.returncode == 0, (p.stdout + p.stderr).strip().replace("\n", " | ")[:200]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--load", choices=list(LOAD_CODE), default="compute")
    ap.add_argument("--caps", type=float, nargs="+", default=[0.8, 0.6, 0.45, 0.35],
                    help="caps as fractions of the DEFAULT limit; clipped to the card's allowed range")
    ap.add_argument("--hold-s", type=float, default=15.0)
    ap.add_argument("--hz", type=float, default=20.0)
    ap.add_argument("--out", default="probe")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    if subprocess.run(["sudo", "-n", "true"]).returncode != 0:
        sys.exit("sudo needs a password here -- this probe needs passwordless sudo")

    pynvml.nvmlInit()
    h = pynvml.nvmlDeviceGetHandleByIndex(a.gpu)
    name = pynvml.nvmlDeviceGetName(h); name = name.decode() if isinstance(name, bytes) else name
    lo, hi = (v / 1000 for v in pynvml.nvmlDeviceGetPowerManagementLimitConstraints(h))
    default_w = pynvml.nvmlDeviceGetPowerManagementDefaultLimit(h) / 1000
    clocks = sorted(pynvml.nvmlDeviceGetSupportedGraphicsClocks(
        h, pynvml.nvmlDeviceGetSupportedMemoryClocks(h)[0]))
    lock_mhz = clocks[len(clocks) // 2]               # a mid-range supported clock
    caps_w = sorted({max(lo, min(hi, round(f * default_w))) for f in a.caps}, reverse=True)
    print(f"GPU {a.gpu}: {name}  limit range {lo:.0f}-{hi:.0f} W, default {default_w:.0f} W", flush=True)
    print(f"cap ladder (W): {caps_w}   clock lock: {lock_mhz} MHz   load: {a.load}", flush=True)

    # Plan: (label, command-args or None, hold seconds). None = just hold.
    plan = [("baseline", None, a.hold_s)]
    plan += [(f"cap_{c:.0f}W", ["-i", str(a.gpu), "-pl", f"{c:.0f}"], a.hold_s) for c in caps_w]
    plan += [("release", ["-i", str(a.gpu), "-pl", f"{default_w:.0f}"], a.hold_s)]
    plan += [(f"lock_{lock_mhz}MHz", ["-i", str(a.gpu), "-lgc", f"{lock_mhz},{lock_mhz}"], a.hold_s)]
    plan += [("unlock", ["-i", str(a.gpu), "-rgc"], a.hold_s)]

    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(a.gpu), CUDA_DEVICE_ORDER="PCI_BUS_ID")
    load = subprocess.Popen([sys.executable, "-c", LOAD_CODE[a.load]], env=env)
    time.sleep(8)                                      # let the load reach steady state

    samples, cmds = [], []
    try:
        for label, args, hold in plan:
            t_cmd = time.time()
            ok, msg = smi(*args) if args else (True, "")
            cmds.append({"t": t_cmd, "step": label, "ok": ok, "msg": msg})
            print(f"  {label:14s} {'ok' if ok else 'FAILED: ' + msg}", flush=True)
            t_end = t_cmd + hold
            while time.time() < t_end:                 # sample this step at ~hz
                t = time.time()
                th = int(_THROTTLE(h)) if _THROTTLE else -1
                samples.append({"t": t, "step": label,
                                "w": pynvml.nvmlDeviceGetPowerUsage(h) / 1000,
                                "limit_w": pynvml.nvmlDeviceGetEnforcedPowerLimit(h) / 1000,
                                "sm_mhz": pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM),
                                "pcap": int(th >= 0 and bool(th & BIT_POWER_CAP))})
                time.sleep(max(0.0, 1 / a.hz - (time.time() - t)))
    finally:
        load.kill()
        smi("-i", str(a.gpu), "-pl", f"{default_w:.0f}"); smi("-i", str(a.gpu), "-rgc")   # always reset

    with open(os.path.join(a.out, "samples.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(samples[0])); w.writeheader(); w.writerows(samples)
    with open(os.path.join(a.out, "commands.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cmds[0])); w.writeheader(); w.writerows(cmds)

    # ---- summary --------------------------------------------------------------
    base = [s["w"] for s in samples if s["step"] == "baseline"]
    base_w = sum(base) / len(base)
    print(f"\nunconstrained power under '{a.load}' load: {base_w:.0f} W ({100 * base_w / default_w:.0f} % of default limit)")
    print(f"{'step':14s} {'limit':>6s} {'mean W':>7s} {'max W':>6s} {'2nd-half W':>10s} {'pcap%':>6s} {'SM MHz':>7s} {'speed s':>8s}  verdict")
    cap_ok, clock_ok = True, True
    for c in cmds:
        ss = [s for s in samples if s["step"] == c["step"]]
        if not ss:
            continue
        half = ss[len(ss) // 2:]
        m = sum(s["w"] for s in ss) / len(ss); m2 = sum(s["w"] for s in half) / len(half)
        lim = ss[-1]["limit_w"]
        verdict, speed = "", ""
        if c["step"].startswith("cap_"):
            cap = float(c["step"][4:-1])
            if base_w <= cap * 1.02:
                verdict = "not binding (load draws less than the cap)"
            else:
                hit = next((s["t"] for s in ss if s["w"] <= cap * 1.05), None)
                speed = f"{hit - c['t']:.2f}" if hit else ">hold"
                good = m2 <= cap * 1.05 and abs(lim - cap) < 1
                cap_ok &= good
                verdict = "cap holds" if good else "CAP IGNORED"
        if c["step"].startswith("lock_"):
            mhz = sum(s["sm_mhz"] for s in half) / len(half)
            good = mhz <= lock_mhz + 15
            clock_ok &= good
            verdict = "lock holds" if good else "LOCK IGNORED"
        if not c["ok"]:
            verdict = "command refused: " + c["msg"]
            cap_ok &= not c["step"].startswith("cap_"); clock_ok &= not c["step"].startswith("lock_")
        print(f"{c['step']:14s} {lim:6.0f} {m:7.1f} {max(s['w'] for s in ss):6.0f} {m2:10.1f} "
              f"{100 * sum(s['pcap'] for s in ss) / len(ss):6.1f} {sum(s['sm_mhz'] for s in ss) / len(ss):7.0f} {speed:>8s}  {verdict}")
    print(f"\nRESULT  power cap: {'WORKS' if cap_ok else 'DOES NOT WORK'}   clock lock: {'WORKS' if clock_ok else 'DOES NOT WORK'}")
    print(f"files: {a.out}/samples.csv, {a.out}/commands.csv")


if __name__ == "__main__":
    main()
