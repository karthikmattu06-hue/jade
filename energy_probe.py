#!/usr/bin/env python3
# ======================================================================
#  JADE  E0.1  --  joules and watts per PREFILL token and per DECODE token
# ======================================================================
#
#  WHAT THIS MEASURES AND WHY
#  --------------------------
#  An LLM answers a request in two phases that behave nothing alike:
#
#    PREFILL  reads your whole prompt at once. COMPUTE-bound: one big dense
#             matrix multiply, high power, short burst. Degrading it hurts
#             time-to-first-token (TTFT).
#
#    DECODE   writes the answer one token at a time. MEMORY-BANDWIDTH-bound:
#             every step re-reads all the model weights out of HBM to produce
#             one token per sequence. Lower power, long duration. Degrading it
#             hurts time-per-output-token (TPOT).
#
#  If those phases have different energy signatures, then splitting them onto
#  different workers (disaggregated serving) hands you a power-management
#  control surface a monolithic server does not have. That is the Jade thesis.
#
#  ---------------------------------------------------------------------
#  WHY THIS VERSION IS DESIGNED DIFFERENTLY FROM v2  -- READ THIS
#  ---------------------------------------------------------------------
#  v2 measured a grid of prompt/output length mixes and solved
#
#      E = a*(prefill tokens) + b*(decode tokens) + c
#
#  by least squares, the way you would separate two appliances sharing one
#  electricity meter. Re-analysing the v2 data exposed a flaw in that plan,
#  and it is worth understanding because it dictates this file's structure.
#
#  Refit v2's own numbers with the constant term left free, and the fit says
#  the baseline power is about 217 W. The meter says parked power is 34 W.
#  A six-fold disagreement is not a rounding error -- it is the design telling
#  you something. What it is telling you is this: inside every v2 window the
#  GPU was pinned busy, so every configuration looked nearly identical to the
#  power meter (194-247 W). Total energy per window was therefore almost
#  exactly (a constant) x (window duration), and window duration barely varied.
#  The regression was mostly fitting window length, and the token coefficients
#  were fighting over the ~30 W of variation left over.
#
#  So this version stops leaning on the regression. Instead:
#
#    * PRIMARY results are measured DIRECTLY, by division, from windows that
#      are already almost pure. A window with output length 1 is ~99.5% prefill
#      work -- just divide its energy by its prefill tokens. No model needed.
#      A batch sweep at long output length does the same for decode.
#    * The regression survives only as a CROSS-CHECK, and it now ships with a
#      validity gate: the script fits the baseline power as a free parameter
#      and compares it to the measured parked power. If those two disagree,
#      the script says so loudly and tells you not to quote the coefficients.
#      A good-looking R-squared is not permission to trust a fit.
#    * A BATCH SWEEP is added. Decode reads the whole model per step no matter
#      how many sequences are in the batch, so joules per decode token should
#      fall roughly as 1/batch until compute takes over. That produces a curve
#      you can see rather than a coefficient you have to believe -- and batch
#      size is a real scheduler lever, so the curve is directly useful later.
#
#  WHAT ELSE IS NEW SINCE v2
#  -------------------------
#   1. Throttle REASONS are sampled, not just clock speed. NVML says outright
#      whether the driver slowed the GPU for the power cap or for heat.
#   2. A cooldown before every window. In v2 temperature ramped 40 -> 62 C
#      across the session, so "hot" and "late in the shuffle" were the same
#      variable and could not be separated.
#   3. generate()-call count is available as a regressor, so per-call overhead
#      has somewhere to live other than being blamed on tokens.
#   4. Prefill cost is reported per prompt length. (Expect it to be flat for a
#      1.5B model at these lengths -- attention is only ~11% of the FLOPs at
#      n=2048, so the O(n^2) term has not bitten yet. It should appear on 7B
#      and at longer prompts.)
#   5. Per-window power / clock / throttle time series are written to disk.
#   6. Parked power (engine loaded, zero requests) is measured three times --
#      cold, mid, hot -- because it is the floor draining can reach without
#      tearing the pod down, and leakage current makes it temperature-dependent.
#
#  STALE-FILE CHECK: the banner prints PROBE_VERSION. If the .out file does not
#  say E0.1 (v3.1), an old copy is on the cluster and nothing below is valid.
# ======================================================================

from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import threading
import time
from dataclasses import dataclass, asdict

import numpy as np

PROBE_VERSION = "E0.1 (v3.1)"


# ======================================================================
#  1.  COMMAND-LINE ARGUMENTS
# ======================================================================
# Everything tunable lives here, so you never edit the body to change an
# experiment. argparse also gives a free stale-file detector: pass a flag that
# only exists in this version and an old copy dies with "unrecognized
# arguments" instead of silently producing plausible-looking wrong numbers.

def parse_args():
    p = argparse.ArgumentParser(
        description=f"Jade E0.1 energy probe {PROBE_VERSION}")

    p.add_argument("--model", default=os.environ.get(
        "JADE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct"))

    # --- phase A: the mix grid ------------------------------------------
    # Output length 1 makes a window that is essentially pure prefill; that is
    # where the direct prefill measurement comes from. The longer output
    # lengths are kept because they give the cross-check regression something
    # to work with and because mean power per mix is itself a result.
    p.add_argument("--prompt-lens", type=int, nargs="+", default=[128, 512, 2048])
    p.add_argument("--output-lens", type=int, nargs="+", default=[1, 64, 256])
    p.add_argument("--batch", type=int, default=8)

    # --- phase B: the batch sweep ---------------------------------------
    # Short prompt, long output: ~97% of the energy here is decode, so dividing
    # gives joules per decode token almost directly. Sweeping batch traces out
    # how that cost falls as the per-step weight read is amortised over more
    # sequences.
    p.add_argument("--batch-sweep", type=int, nargs="+",
                   default=[1, 2, 4, 8, 16, 32])
    p.add_argument("--sweep-prompt", type=int, default=128)
    p.add_argument("--sweep-output", type=int, default=256)

    p.add_argument("--reps", type=int, default=3,
                   help="repetitions of every configuration, for error bars")
    p.add_argument("--seed", type=int, default=0)

    # --- measurement window ---------------------------------------------
    # The NVML energy counter only ticks every ~100 ms. A window shorter than
    # that reads zero joules -- the bug that ruined v1. So we repeat generate()
    # inside a window until --min-seconds have passed and counter granularity
    # becomes a rounding error.
    p.add_argument("--min-seconds", type=float, default=3.0)
    p.add_argument("--max-inner", type=int, default=400)
    p.add_argument("--sample-hz", type=float, default=20.0)

    # --- cooldown --------------------------------------------------------
    p.add_argument("--cool-to", type=float, default=45.0,
                   help="wait until the GPU is at or below this temperature "
                        "before each window; set very high to disable")
    p.add_argument("--cool-max-s", type=float, default=30.0,
                   help="stop waiting after this long and proceed anyway")

    p.add_argument("--park-seconds", type=float, default=8.0)

    # --- validity gate ----------------------------------------------------
    p.add_argument("--park-tolerance-w", type=float, default=20.0,
                   help="if the regression's fitted baseline power differs "
                        "from measured parked power by more than this, the "
                        "cross-check fit is declared untrustworthy")

    p.add_argument("--out-prefix", default=os.environ.get("JADE_OUT", "jade_e0"))
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--gpu-mem-util", type=float, default=0.85)
    return p.parse_args()


# ======================================================================
#  2.  NVML  --  the GPU's own instrumentation
# ======================================================================
# NVML is the C library behind `nvidia-smi`. nvidia-smi is a program that calls
# NVML and prints text; pynvml is a Python binding that calls the same NVML
# directly. Same numbers, no subprocess, no text parsing, and we can poll at
# 20 Hz. Reading is unprivileged. WRITING -- setting a power cap, locking
# clocks -- needs root, which is the conversation to have with RC for Phase 3.

import pynvml  # noqa: E402

# --- API compatibility shim -------------------------------------------
# NVIDIA renamed the throttle API: older pynvml has
# nvmlDeviceGetCurrentClocksThrottleReasons, newer has ...ClocksEventReasons.
# Bind whichever exists so a pynvml upgrade cannot kill a run mid-flight.
_THROTTLE_FN = (getattr(pynvml, "nvmlDeviceGetCurrentClocksThrottleReasons", None)
                or getattr(pynvml, "nvmlDeviceGetCurrentClocksEventReasons", None))


def _bit(new_name: str, old_name: str, default: int) -> int:
    v = getattr(pynvml, new_name, None)
    if v is None:
        v = getattr(pynvml, old_name, None)
    return default if v is None else int(v)


# Each bit means "this is currently holding the clocks down". The two that
# matter for Jade: SW_POWER_CAP (limited to stay under the power cap) and the
# THERMAL bits (limited because the die is hot). Telling them apart is the
# difference between "the cap binds" as an inference and as a measurement.
BIT_POWER_CAP = _bit("nvmlClocksEventReasonSwPowerCap",
                     "nvmlClocksThrottleReasonSwPowerCap", 0x4)
BIT_HW_SLOWDOWN = _bit("nvmlClocksEventReasonHwSlowdown",
                       "nvmlClocksThrottleReasonHwSlowdown", 0x8)
BIT_SW_THERMAL = _bit("nvmlClocksEventReasonSwThermalSlowdown",
                      "nvmlClocksThrottleReasonSwThermalSlowdown", 0x20)
BIT_HW_THERMAL = _bit("nvmlClocksEventReasonHwThermalSlowdown",
                      "nvmlClocksThrottleReasonHwThermalSlowdown", 0x40)
BIT_HW_POWERBRAKE = _bit("nvmlClocksEventReasonHwPowerBrakeSlowdown",
                         "nvmlClocksThrottleReasonHwPowerBrakeSlowdown", 0x80)


def read_throttle(h) -> int:
    if _THROTTLE_FN is None:
        return 0
    try:
        return int(_THROTTLE_FN(h))
    except pynvml.NVMLError:
        return 0


def energy_joules(handles) -> float:
    """
    Joules drawn since the driver loaded, summed over GPUs.

    nvmlDeviceGetTotalEnergyConsumption returns MILLIjoules from a hardware
    accumulator. Far better than integrating your own power samples: the
    counter cannot miss a spike between two samples. The counter is the primary
    measurement; the sampler below is a cross-check and gives us shape.
    """
    return sum(pynvml.nvmlDeviceGetTotalEnergyConsumption(h)
               for h in handles) / 1000.0


def gpu_static_info(h) -> dict:
    """
    One-time facts. The power-limit constraints matter for Phase 3: they are
    the range of caps you are allowed to set, so they bound how deep a
    "cap the GPU" demand-response lever could possibly go.
    """
    info = {}

    def try_(key, fn, scale=1.0):
        try:
            info[key] = fn() * scale
        except Exception:
            info[key] = None

    try:
        name = pynvml.nvmlDeviceGetName(h)
        info["name"] = name.decode() if isinstance(name, bytes) else str(name)
    except Exception:
        info["name"] = None
    try_("power_limit_w", lambda: pynvml.nvmlDeviceGetPowerManagementLimit(h), 1e-3)
    try_("power_limit_default_w",
         lambda: pynvml.nvmlDeviceGetPowerManagementDefaultLimit(h), 1e-3)
    try_("power_limit_enforced_w",
         lambda: pynvml.nvmlDeviceGetEnforcedPowerLimit(h), 1e-3)
    try:
        lo, hi = pynvml.nvmlDeviceGetPowerManagementLimitConstraints(h)
        info["power_limit_min_w"], info["power_limit_max_w"] = lo / 1e3, hi / 1e3
    except Exception:
        info["power_limit_min_w"] = info["power_limit_max_w"] = None
    try_("sm_clock_max_mhz",
         lambda: pynvml.nvmlDeviceGetMaxClockInfo(h, pynvml.NVML_CLOCK_SM))
    info["throttle_api"] = _THROTTLE_FN.__name__ if _THROTTLE_FN else None
    return info


# ======================================================================
#  3.  THE SAMPLER  --  a background thread watching the GPU
# ======================================================================
# The energy counter gives the total; the sampler gives the shape -- how power
# rose, whether the clock fell, and why. It runs on its own thread so it keeps
# sampling while the main thread is blocked inside generate().

class PowerSampler:
    def __init__(self, handles, hz: float = 20.0):
        self.handles = handles
        self.period = 1.0 / hz
        self._stop = threading.Event()
        self._thread = None
        self.t, self.w, self.mhz, self.temp, self.flags = [], [], [], [], []

    def _loop(self):
        t0 = time.perf_counter()
        h0 = self.handles[0]
        while not self._stop.is_set():
            try:
                w = sum(pynvml.nvmlDeviceGetPowerUsage(h)
                        for h in self.handles) / 1000.0
                self.t.append(time.perf_counter() - t0)
                self.w.append(w)
                self.mhz.append(pynvml.nvmlDeviceGetClockInfo(h0, pynvml.NVML_CLOCK_SM))
                self.temp.append(pynvml.nvmlDeviceGetTemperature(
                    h0, pynvml.NVML_TEMPERATURE_GPU))
                self.flags.append(read_throttle(h0))
            except pynvml.NVMLError:
                pass
            # Event.wait rather than sleep, so stop() returns promptly.
            self._stop.wait(self.period)

    def start(self):
        self._stop.clear()
        for lst in (self.t, self.w, self.mhz, self.temp, self.flags):
            lst.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def frac_with(self, bit: int) -> float:
        if not self.flags:
            return 0.0
        return sum(1 for f in self.flags if f & bit) / len(self.flags)

    def summary(self) -> dict:
        if not self.w:
            return dict(mean_w=float("nan"), peak_w=float("nan"), n_samples=0,
                        sm_mhz_mean=float("nan"), sm_mhz_min=0, sm_mhz_end=0,
                        temp_c_end=0, frac_power_capped=0.0, frac_thermal=0.0,
                        frac_hw_slowdown=0.0)
        # The sampler starts before the first kernel launches, so the first
        # few samples see the idle clock (765 MHz on an A100). A raw minimum
        # reports that and makes every window look throttled -- 37 of 45 did
        # in the first E0.1 run. Clock statistics are therefore taken only
        # over samples where the GPU is working: power >= half the peak.
        thr = 0.5 * max(self.w)
        busy = [m for m, w in zip(self.mhz, self.w) if w >= thr] or self.mhz
        return dict(
            mean_w=statistics.fmean(self.w), peak_w=max(self.w),
            n_samples=len(self.w), sm_mhz_mean=statistics.fmean(busy),
            sm_mhz_min=min(busy), sm_mhz_end=self.mhz[-1],
            temp_c_end=self.temp[-1],
            frac_power_capped=self.frac_with(BIT_POWER_CAP | BIT_HW_POWERBRAKE),
            frac_thermal=self.frac_with(BIT_SW_THERMAL | BIT_HW_THERMAL),
            frac_hw_slowdown=self.frac_with(BIT_HW_SLOWDOWN))

    def trace(self) -> dict:
        return dict(t=[round(x, 4) for x in self.t],
                    w=[round(x, 2) for x in self.w],
                    mhz=list(self.mhz), temp=list(self.temp),
                    flags=list(self.flags))


# ======================================================================
#  4.  RESULT RECORD
# ======================================================================

@dataclass
class Run:
    phase: str              # "grid" or "sweep"
    prompt_tokens: int
    output_tokens: int
    batch: int
    rep: int

    inner_reps: int         # generate() calls inside the timing window
    wall_s: float
    energy_j: float         # hardware counter, total for the window
    idle_j: float           # parked power * wall_s: the baseline removed
    net_j: float            # energy_j - idle_j: the cost of doing the work

    prefill_tokens: int     # P * batch * inner_reps
    decode_tokens: int      # counted from the outputs, not assumed

    mean_w: float
    peak_w: float
    counter_w: float        # energy_j / wall_s; must agree with mean_w
    n_samples: int
    sm_mhz_mean: float
    sm_mhz_min: int
    sm_mhz_end: int
    temp_c_start: int
    temp_c_end: int
    frac_power_capped: float
    frac_thermal: float
    frac_hw_slowdown: float
    cooled_ok: bool

    @property
    def decode_share(self) -> float:
        tot = self.prefill_tokens + self.decode_tokens
        return self.decode_tokens / tot if tot else 0.0


# ======================================================================
#  5.  SYNTHETIC PROMPTS
# ======================================================================

def token_prompt(n_tokens: int, rng: random.Random, vocab_hi: int) -> dict:
    """
    Build a prompt of EXACTLY n_tokens by handing vLLM raw token ids.

    Not real text, because "512 tokens of English" is not a length you can dial
    in -- the tokenizer decides, and you would end up measuring your tokenizer
    rather than the GPU. Ids make prompt length an exact experimental variable.

    Ids come from the middle of the vocabulary (above 1000, below the top 1000)
    to dodge special tokens: BOS/EOS/PAD live at the edges and change how a
    request is handled.

    Randomness is the point, not a shortcut. Repeated text would let the KV
    cache serve a later request from an earlier one's stored keys and values,
    and we would be measuring cache hits instead of prefill. Prefix caching is
    also switched off at the engine; this is belt and braces.
    """
    return {"prompt_token_ids": [rng.randrange(1000, vocab_hi)
                                 for _ in range(n_tokens)]}


# ======================================================================
#  6.  COOLDOWN AND PARKED POWER
# ======================================================================

def cooldown(h, target_c: float, max_s: float) -> tuple[bool, int]:
    """
    Idle the GPU until it drops to target_c, or until max_s elapses.

    Leakage current rises with temperature, so a hot GPU really does draw more
    power doing identical work. In v2 the session heated monotonically from
    40 C to 62 C, which meant "later in the run" and "hotter" were the same
    variable. Equalising the start temperature removes that confound.
    """
    t0 = time.perf_counter()
    while True:
        temp = pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
        if temp <= target_c:
            return True, temp
        if time.perf_counter() - t0 >= max_s:
            return False, temp
        time.sleep(1.0)


def measure_parked(handles, sampler, seconds: float, label: str) -> dict:
    """
    Power with the engine fully loaded and ZERO requests in flight.

    Not the same as "GPU idle": weights are resident, the CUDA context is up,
    vLLM's scheduler is running. This is the floor DRAINING reaches -- stop
    sending a worker requests, let the in-flight ones finish, and this is what
    remains. Going lower means evicting the model and tearing the pod down, a
    much slower and more disruptive lever. So this number sets the depth of the
    cheap lever, and it is one of the headline results of the project.

    Taken three times because leakage means it is not necessarily a constant.
    """
    sampler.start()
    e0 = energy_joules(handles)
    t0 = time.perf_counter()
    time.sleep(seconds)
    wall = time.perf_counter() - t0
    e = energy_joules(handles) - e0
    sampler.stop()
    s = sampler.summary()
    return dict(label=label, wall_s=wall, counter_w=e / wall,
                sampled_w=s["mean_w"], temp_c=s["temp_c_end"],
                sm_mhz=s["sm_mhz_end"])


# ======================================================================
#  7.  ANALYSIS
# ======================================================================

def mean_se(xs: list[float]) -> tuple[float, float]:
    """Mean and standard error. With one sample the error is unknown, not zero."""
    if not xs:
        return float("nan"), float("nan")
    if len(xs) == 1:
        return xs[0], float("nan")
    return statistics.fmean(xs), statistics.stdev(xs) / math.sqrt(len(xs))


def direct_attribution(runs: list[Run], grid_batch: int, iters: int = 3) -> dict:
    """
    PRIMARY RESULT. Joules per token by division, not by fitting.

    The idea: some windows are already almost pure. A window with output
    length 1 spends ~99.5% of its tokens in prefill, so
    net joules / prefill tokens is very nearly the prefill cost -- the only
    contamination is the handful of decode tokens. Likewise a short-prompt,
    long-output window is ~97% decode by energy.

    The small cross-contamination is removed by iterating: estimate prefill
    ignoring decode, use it to clean the decode windows, use that decode number
    to clean the prefill windows, repeat. Three passes is plenty; the
    correction is a few percent and shrinks each time.

    This is deliberately dumber than a regression, and that is the point. It
    has no intercept to soak up energy, no collinearity to worry about, and no
    way to look good while being wrong.
    """
    # Purest prefill windows per prompt length: the smallest decode share.
    grid = [r for r in runs if r.phase == "grid"]
    by_P: dict[int, list[Run]] = {}
    for r in grid:
        by_P.setdefault(r.prompt_tokens, []).append(r)
    pure_pf = {P: [r for r in rs
                   if r.decode_share == min(x.decode_share for x in rs)]
               for P, rs in by_P.items()}
    pf_all = [r for rs in pure_pf.values() for r in rs]

    # Purest decode windows: the largest decode share, AT THE GRID BATCH SIZE
    # ONLY. Joules per decode token depends almost entirely on batch (see the
    # sweep), so mixing batch sizes here would average 0.67 J/token at batch 1
    # with 0.026 at batch 32 into a number that describes nothing. The first
    # E0.1 run had exactly this bug and printed a 74x ratio; the true batch-8
    # figure is ~27x.
    cand = [r for r in runs if r.batch == grid_batch]
    best_share = max(r.decode_share for r in cand)
    pure_dc = [r for r in cand if r.decode_share >= best_share - 1e-9]

    # --- the one small fit we do keep, and why it is safe ---------------
    # Even a pure-prefill window carries per-generate()-call overhead
    # (scheduler, batch setup, CUDA graph launch). Dividing straight through
    # blames that on tokens, which inflates short prompts most: at P=128 each
    # call carries only 1024 prefill tokens to spread the overhead over, at
    # P=2048 it carries 16384. So over the pure-prefill windows ONLY we solve
    #
    #     net J = a*(prefill tokens) + d*(calls)
    #
    # Two parameters, no intercept, no decode term, and the three prompt
    # lengths give three genuinely different tokens-per-call ratios -- so it is
    # well conditioned, unlike the full mixed regression. This is the only
    # place a solver appears in the primary path.
    d_call = 0.0
    if len(pf_all) >= 3 and len({r.prompt_tokens for r in pf_all}) >= 2:
        Apf = np.column_stack([[r.prefill_tokens for r in pf_all],
                               [r.inner_reps for r in pf_all]]).astype(float)
        ypf = np.array([r.net_j for r in pf_all], dtype=float)
        cpf, spf, stpf = fit_scaled(Apf, ypf)
        d_call = float(cpf[1])
        call_fit = dict(a=float(cpf[0]), a_stderr=float(spf[0]),
                        d=float(cpf[1]), d_stderr=float(spf[1]),
                        r2=stpf["r2"], cond=stpf["cond"], n=len(pf_all))
    else:
        call_fit = None

    b = 0.0  # J per decode token, refined each pass
    a_by_P: dict[int, tuple[float, float]] = {}
    for _ in range(iters):
        # Pass 1: prefill cost per prompt length, with the decode tokens and
        # the per-call overhead both removed.
        a_by_P = {}
        for P, rs in pure_pf.items():
            vals = [(r.net_j - b * r.decode_tokens - d_call * r.inner_reps)
                    / r.prefill_tokens for r in rs]
            a_by_P[P] = mean_se(vals)
        a_mean = statistics.fmean(v for v, _ in a_by_P.values())
        # Pass 2: decode cost, with prefill and per-call overhead removed.
        vals = [(r.net_j - a_mean * r.prefill_tokens - d_call * r.inner_reps)
                / r.decode_tokens for r in pure_dc]
        b, b_se = mean_se(vals)

    a_all = [v for v, _ in a_by_P.values()]
    a, a_se = mean_se(a_all) if len(a_all) > 1 else (a_all[0], float("nan"))

    # Purity is reported by ENERGY share, not token share -- a window can be a
    # third prefill by token count and still be 97% decode by joules, and it is
    # the joules that decide how clean the division is.
    def dc_energy_share(r):
        pe, de = a * r.prefill_tokens, b * r.decode_tokens
        return de / (pe + de) if (pe + de) else 0.0

    return dict(
        j_per_prefill_token=dict(value=a, stderr=a_se),
        j_per_decode_token=dict(value=b, stderr=b_se),
        j_per_call=d_call, call_fit=call_fit,
        ratio=b / a if a else float("nan"),
        by_prompt_len={str(P): dict(value=v, stderr=e)
                       for P, (v, e) in sorted(a_by_P.items())},
        prefill_windows=sorted({f"P{r.prompt_tokens}/D{r.output_tokens}"
                                for r in pf_all}),
        decode_windows=sorted({f"P{r.prompt_tokens}/D{r.output_tokens}"
                               f"/b{r.batch}" for r in pure_dc}),
        purity_prefill=1 - max(dc_energy_share(r) for r in pf_all),
        purity_decode=min(dc_energy_share(r) for r in pure_dc),
    )


def decode_vs_batch(runs: list[Run], a_prefill: float) -> dict:
    """
    Joules per decode token at each batch size (phase B).

    A decode step reads every model weight out of HBM once, no matter how many
    sequences are in the batch -- the read is shared. So doubling the batch
    should roughly halve the energy per token, until the batch grows large
    enough that arithmetic rather than memory traffic becomes the limit and the
    curve flattens. Where it flattens is a real scheduling fact: below that
    point, batching is nearly free energy savings; above it, it costs latency
    for nothing.
    """
    out = {}
    for r in [x for x in runs if x.phase == "sweep"]:
        j = (r.net_j - a_prefill * r.prefill_tokens) / r.decode_tokens
        out.setdefault(r.batch, []).append(j)
    return {str(k): dict(value=mean_se(v)[0], stderr=mean_se(v)[1], n=len(v))
            for k, v in sorted(out.items())}


def fit_scaled(A_raw: np.ndarray, y: np.ndarray):
    """
    Least squares with column scaling, returning coefficients and std errors.

    Why scale? The columns differ wildly in magnitude -- prefill tokens run to
    ~180,000 while call counts are single digits. Fed straight to a solver that
    is badly conditioned: precision is lost and, worse, the standard errors come
    out meaningless. Dividing each column by its own RMS puts them on a
    comparable footing; we solve there and undo the scaling afterwards. The
    fitted model is identical; the error bars become trustworthy.
    """
    scale = np.sqrt((A_raw ** 2).mean(axis=0))
    scale[scale == 0] = 1.0
    A = A_raw / scale
    coef_s, *_ = np.linalg.lstsq(A, y, rcond=None)
    coef = coef_s / scale
    resid = y - A_raw @ coef
    dof = max(len(y) - A.shape[1], 1)
    sigma2 = float((resid ** 2).sum() / dof)
    try:
        stderr = np.sqrt(np.diag(sigma2 * np.linalg.inv(A.T @ A))) / scale
    except np.linalg.LinAlgError:
        stderr = np.full(len(coef), float("nan"))
    ss_res = float((resid ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return coef, stderr, dict(
        r2=1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        rmse_j=float(np.sqrt((resid ** 2).mean())),
        rmse_pct=100.0 * float(np.sqrt((resid ** 2).mean())) / float(y.mean()),
        cond=float(np.linalg.cond(A)), dof=dof)


def crosscheck_fit(runs: list[Run], parked_w: float, tol_w: float) -> dict:
    """
    The regression, demoted to a cross-check, with a validity gate.

    Model:  total joules = a*prefill + b*decode + d*calls + f*seconds

    The last term is the point. f is the baseline power the data implies, in
    watts, fitted freely. We already MEASURED that number with the engine
    parked. If the fit wants a baseline far from the measured one, the design
    has too little leverage -- the configurations all look alike to the meter,
    and the token coefficients are just absorbing whatever is left over. On the
    v2 data this test fails hard: fitted 217 W against a measured 34 W.

    Passing the gate does not prove the coefficients are right. Failing it
    proves they are not worth quoting.
    """
    y = np.array([r.energy_j for r in runs], dtype=float)
    A = np.column_stack([
        [r.prefill_tokens for r in runs],
        [r.decode_tokens for r in runs],
        [r.inner_reps for r in runs],
        [r.wall_s for r in runs],
    ]).astype(float)
    coef, se, st = fit_scaled(A, y)
    fitted_park = float(coef[3])
    gap = abs(fitted_park - parked_w)
    return dict(a=float(coef[0]), a_se=float(se[0]),
                b=float(coef[1]), b_se=float(se[1]),
                d=float(coef[2]), d_se=float(se[2]),
                fitted_baseline_w=fitted_park, measured_parked_w=parked_w,
                baseline_gap_w=gap, trustworthy=bool(gap <= tol_w), **st)


# ======================================================================
#  8.  MAIN
# ======================================================================

def main():
    args = parse_args()
    rng = random.Random(args.seed)
    print(f"[jade] PROBE_VERSION = {PROBE_VERSION}", flush=True)

    pynvml.nvmlInit()
    n_gpu = pynvml.nvmlDeviceGetCount()
    handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(n_gpu)]
    print(f"[jade] NVML sees {n_gpu} GPU(s); "
          f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}",
          flush=True)

    static = gpu_static_info(handles[0])
    print(f"[jade] {static['name']}  cap {static['power_limit_w']:.0f} W "
          f"(settable {static['power_limit_min_w']:.0f}-"
          f"{static['power_limit_max_w']:.0f} W)  "
          f"base SM {static['sm_clock_max_mhz']} MHz", flush=True)
    if static["throttle_api"] is None:
        print("[jade] WARNING: no throttle-reason API on this driver; "
              "throttle fractions will read 0.", flush=True)

    from vllm import LLM, SamplingParams

    print(f"[jade] loading {args.model} ...", flush=True)
    t0 = time.perf_counter()
    llm = LLM(model=args.model, max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_mem_util,
              # Prefix caching OFF: with it on, a repeated prompt prefix is
              # served from cached keys and values and skips prefill entirely.
              # We would be measuring the cache, not the phase.
              enable_prefix_caching=False, disable_log_stats=True,
              seed=args.seed)
    print(f"[jade] loaded in {time.perf_counter() - t0:.1f}s", flush=True)

    vocab_hi = max(llm.get_tokenizer().vocab_size - 1000, 2000)
    sampler = PowerSampler(handles, hz=args.sample_hz)

    # --- warm-up -------------------------------------------------------
    # The first generate() is not representative of anything: vLLM JIT-compiles
    # kernels with torch.compile, captures CUDA graphs, and allocates its KV
    # cache pool on first use. Those costs are one-off and large. Spend them
    # here, outside the meter, or they land entirely in whichever configuration
    # happened to be measured first and corrupt exactly one row with no trace.
    print("[jade] warm-up ...", flush=True)
    warm = SamplingParams(temperature=0.0, max_tokens=16, min_tokens=16,
                          ignore_eos=True, detokenize=False)
    llm.generate([token_prompt(256, rng, vocab_hi)
                  for _ in range(max(args.batch, max(args.batch_sweep)))],
                 warm, use_tqdm=False)

    parked = [measure_parked(handles, sampler, args.park_seconds, "cold")]
    idle_w = parked[0]["counter_w"]
    print(f"[jade] parked (cold): counter {idle_w:.1f} W | sampled "
          f"{parked[0]['sampled_w']:.1f} W | {parked[0]['temp_c']} C  "
          f"(the two W figures should agree within ~1 W)", flush=True)

    # --- the plan ------------------------------------------------------
    # Phase A is the mix grid; phase B is the batch sweep. Both are shuffled
    # together, because anything that drifts across the session -- die
    # temperature, a neighbour's job, clock behaviour -- would otherwise line
    # up with the running order and masquerade as a real effect. Shuffling
    # turns systematic bias into noise the error bars can honestly report.
    plan = [("grid", P, D, args.batch, r)
            for r in range(args.reps)
            for P in args.prompt_lens for D in args.output_lens]
    plan += [("sweep", args.sweep_prompt, args.sweep_output, bs, r)
             for r in range(args.reps) for bs in args.batch_sweep]
    rng.shuffle(plan)
    print(f"[jade] {len(plan)} windows "
          f"(grid {len(args.prompt_lens) * len(args.output_lens) * args.reps}, "
          f"sweep {len(args.batch_sweep) * args.reps}), each >= "
          f"{args.min_seconds}s, cooled to <= {args.cool_to:.0f} C", flush=True)

    runs: list[Run] = []
    traces: dict[str, dict] = {}

    for i, (phase, P, D, bs, rep) in enumerate(plan, start=1):
        cooled_ok, temp_start = cooldown(handles[0], args.cool_to, args.cool_max_s)

        # A small pool of distinct prompt sets, cycled across inner reps, so
        # repeats cannot hit any residual caching.
        pool = [[token_prompt(P, rng, vocab_hi) for _ in range(bs)]
                for _ in range(4)]
        sp = SamplingParams(
            temperature=0.0,   # greedy: no sampling work, fully repeatable
            max_tokens=D, min_tokens=D,  # with ignore_eos, forces exactly D
            ignore_eos=True,   # do not stop early on an EOS token
            detokenize=False,  # skip CPU text conversion we never read
        )

        # ---- measurement window ----------------------------------------
        sampler.start()
        e_start = energy_joules(handles)
        t_start = time.perf_counter()
        inner, decode_tokens = 0, 0
        while True:
            outs = llm.generate(pool[inner % len(pool)], sp, use_tqdm=False)
            decode_tokens += sum(len(o.outputs[0].token_ids) for o in outs)
            inner += 1
            if (time.perf_counter() - t_start >= args.min_seconds
                    or inner >= args.max_inner):
                break
        wall = time.perf_counter() - t_start
        e_used = energy_joules(handles) - e_start
        sampler.stop()
        # -----------------------------------------------------------------

        s = sampler.summary()
        run = Run(phase=phase, prompt_tokens=P, output_tokens=D, batch=bs,
                  rep=rep, inner_reps=inner, wall_s=wall, energy_j=e_used,
                  idle_j=idle_w * wall, net_j=e_used - idle_w * wall,
                  prefill_tokens=P * bs * inner, decode_tokens=decode_tokens,
                  counter_w=e_used / wall, temp_c_start=temp_start,
                  cooled_ok=cooled_ok, **s)
        runs.append(run)
        traces[f"{i:02d}_{phase}_P{P}_D{D}_b{bs}_rep{rep}"] = sampler.trace()

        flag = ""
        if run.frac_power_capped > 0.05:
            flag += f"  CAP {run.frac_power_capped * 100:.0f}%"
        if run.frac_thermal > 0.05:
            flag += f"  THERM {run.frac_thermal * 100:.0f}%"
        if not cooled_ok:
            flag += "  HOT"
        print(f"[jade] {i:2d}/{len(plan)} {phase:5s} P={P:<5d} D={D:<4d} "
              f"b={bs:<3d} x{inner:<3d} {wall:5.2f}s  net {run.net_j:7.1f} J  "
              f"{run.mean_w:5.1f} W  SM {run.sm_mhz_min}-{run.sm_mhz_end}  "
              f"{temp_start}->{run.temp_c_end}C{flag}", flush=True)

        if i == len(plan) // 2:
            parked.append(measure_parked(handles, sampler,
                                         args.park_seconds, "mid"))

    parked.append(measure_parked(handles, sampler, args.park_seconds, "hot"))

    # ==================================================================
    #  ANALYSIS
    # ==================================================================
    direct = direct_attribution(runs, args.batch)
    a = direct["j_per_prefill_token"]["value"]
    b = direct["j_per_decode_token"]["value"]
    sweep = decode_vs_batch(runs, a)
    cross = crosscheck_fit(runs, idle_w, args.park_tolerance_w)

    grid = [r for r in runs if r.phase == "grid"]
    pf_dom = min(grid, key=lambda r: r.decode_share)
    dc_dom = max(grid, key=lambda r: r.decode_share)
    n_capped = sum(1 for r in runs if r.frac_power_capped > 0.05)
    n_therm = sum(1 for r in runs if r.frac_thermal > 0.05)
    n_neg = sum(1 for r in runs if r.net_j < 0)
    n_hot = sum(1 for r in runs if not r.cooled_ok)

    L = "=" * 70
    print("\n" + L)
    print(f"probe version         : {PROBE_VERSION}")
    print(f"model                 : {args.model}")
    print(f"GPU / cap             : {static['name']} / "
          f"{static['power_limit_w']:.0f} W "
          f"(min settable {static['power_limit_min_w']:.0f} W)")
    print("-" * 70)
    print("PARKED POWER  (engine loaded, zero requests) -- the drain floor")
    for p in parked:
        print(f"   {p['label']:<5s} {p['counter_w']:7.2f} W counter | "
              f"{p['sampled_w']:7.2f} W sampled | {p['temp_c']:3d} C")
    print("-" * 70)
    print("PRIMARY  -- direct division, no model fitted")
    print(f"   purity: prefill windows {direct['purity_prefill'] * 100:.1f}% "
          f"prefill | decode windows {direct['purity_decode'] * 100:.1f}% decode")
    print(f"   J / prefill token  : {a:10.5f}  +/- "
          f"{direct['j_per_prefill_token']['stderr']:.5f}")
    print(f"   J / decode  token  : {b:10.5f}  +/- "
          f"{direct['j_per_decode_token']['stderr']:.5f}   "
          f"(at batch {args.batch})")
    print(f"   decode / prefill   : {direct['ratio']:10.1f}x per token")
    if direct.get("call_fit"):
        cf = direct["call_fit"]
        print(f"   J / generate() call: {cf['d']:10.3f}  +/- {cf['d_stderr']:.3f}"
              f"   (removed above; from a 2-parameter fit on the")
        print(f"                        {cf['n']} pure-prefill windows, "
              f"R2 {cf['r2']:.3f}, cond {cf['cond']:.1f})")
    print("   J / prefill token by prompt length "
          "(flat is expected for 1.5B at <=2048)")
    for P, v in direct["by_prompt_len"].items():
        print(f"      P={P:<6s} {v['value']:10.5f}  +/- {v['stderr']:.5f}")
    print("-" * 70)
    print(f"BATCH SWEEP -- J per decode token vs batch "
          f"(P={args.sweep_prompt}, D={args.sweep_output})")
    print("   expect ~1/batch while memory-bound, then a flattening")
    base = None
    for k, v in sweep.items():
        if base is None:
            base = v["value"]
        print(f"      batch {k:<4s} {v['value']:10.5f}  +/- {v['stderr']:.5f}"
              f"   ({base / v['value'] if v['value'] else float('nan'):5.2f}x "
              f"better than batch {list(sweep)[0]})")
    print("-" * 70)
    print("POWER")
    print(f"   prefill-dominated  : {pf_dom.mean_w:8.1f} W  "
          f"(P={pf_dom.prompt_tokens}, D={pf_dom.output_tokens})")
    print(f"   decode-dominated   : {dc_dom.mean_w:8.1f} W  "
          f"(P={dc_dom.prompt_tokens}, D={dc_dom.output_tokens})")
    print(f"   parked             : {idle_w:8.1f} W")
    print(f"   DRAIN depth        : {pf_dom.mean_w - idle_w:8.1f} W prefill | "
          f"{dc_dom.mean_w - idle_w:.1f} W decode")
    if static.get("power_limit_w"):
        print(f"   CAP headroom       : "
              f"{static['power_limit_w'] - pf_dom.mean_w:8.1f} W prefill | "
              f"{static['power_limit_w'] - dc_dom.mean_w:.1f} W decode  "
              f"(a cap above the decode figure is a no-op for decode)")
    print("-" * 70)
    print("THROTTLE")
    print(f"   power-capped windows   : {n_capped:3d} / {len(runs)}")
    print(f"   thermal-limited windows: {n_therm:3d} / {len(runs)}")
    print(f"   cooldown missed target : {n_hot:3d} / {len(runs)}")
    print("-" * 70)
    print("CROSS-CHECK REGRESSION  (secondary; gated)")
    print(f"   a prefill {cross['a']:.5f} +/- {cross['a_se']:.5f} | "
          f"b decode {cross['b']:.5f} +/- {cross['b_se']:.5f} | "
          f"ratio {cross['b'] / cross['a'] if cross['a'] else float('nan'):.1f}x")
    print(f"   R2 {cross['r2']:.4f}   RMSE {cross['rmse_j']:.1f} J "
          f"({cross['rmse_pct']:.1f}%)   cond {cross['cond']:.1f}")
    print(f"   fitted baseline {cross['fitted_baseline_w']:.1f} W  vs  "
          f"measured parked {cross['measured_parked_w']:.1f} W  "
          f"(gap {cross['baseline_gap_w']:.1f} W)")
    if cross["trustworthy"]:
        print("   GATE PASSED: the fit recovers the measured baseline, so its")
        print("   coefficients are a genuine independent check on the direct")
        print("   numbers above. They should agree; if they do not, say so.")
    else:
        print("   *** GATE FAILED ***  The fit cannot recover the baseline it")
        print("   should already know. That means the configurations look too")
        print("   alike to the power meter and the coefficients are absorbing")
        print("   leftovers. DO NOT QUOTE THEM. The direct numbers stand.")
        print("   Fix by widening the design: vary request rate so the GPU is")
        print("   not pinned busy in every window.")
    print("-" * 70)
    print(f"negative net-energy rows : {n_neg:3d}   (must be 0)")
    print(L + "\n", flush=True)

    out = dict(
        probe_version=PROBE_VERSION, model=args.model, reps=args.reps,
        seed=args.seed, batch_grid=args.batch, batch_sweep=args.batch_sweep,
        min_seconds=args.min_seconds, cool_to=args.cool_to,
        prefix_caching=False, shuffled=True, gpus_measured=n_gpu, gpu=static,
        parked=parked, idle_w_used=idle_w,
        direct=direct, decode_vs_batch=sweep, crosscheck=cross,
        w_prefill_dominated=pf_dom.mean_w, w_decode_dominated=dc_dom.mean_w,
        drain_depth_prefill_w=pf_dom.mean_w - idle_w,
        drain_depth_decode_w=dc_dom.mean_w - idle_w,
        n_windows_power_capped=n_capped, n_windows_thermal=n_therm,
        runs=[asdict(r) for r in runs])
    with open(f"{args.out_prefix}.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"[jade] wrote {args.out_prefix}.json", flush=True)
    with open(f"{args.out_prefix}_traces.json", "w") as f:
        json.dump(traces, f)
    print(f"[jade] wrote {args.out_prefix}_traces.json", flush=True)

    make_plots(args, runs, traces, direct, sweep, cross, idle_w, static)


# ======================================================================
#  9.  PLOTS
# ======================================================================
# Colours are Okabe-Ito, chosen to stay distinguishable under the common forms
# of colour blindness; blue and orange also survive greyscale printing, which
# matters for a paper appendix. Every series is direct-labelled or legended, so
# identity never rests on colour alone.

C_PREFILL = "#0072B2"   # blue
C_DECODE = "#E69F00"    # orange
C_OK = "#009E73"        # green      -- running at full clock
C_CAPPED = "#D55E00"    # vermillion -- held back by the power cap
C_INK, C_MUTED = "#333333", "#888888"


def make_plots(args, runs, traces, direct, sweep, cross, idle_w, static):
    import matplotlib
    matplotlib.use("Agg")              # no display on a compute node
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.dpi": 130, "font.size": 8, "axes.grid": True,
        "grid.color": "#e8e8e8", "grid.linewidth": 0.6,
        "axes.axisbelow": True,        # data draws on top of the grid
        "axes.edgecolor": "#cccccc", "axes.labelcolor": C_INK,
        "text.color": C_INK, "xtick.color": C_MUTED, "ytick.color": C_MUTED,
        "legend.frameon": False,
    })
    fig, ax = plt.subplots(2, 3, figsize=(13.5, 7.4))

    # -- (0,0) prefill cost by prompt length -----------------------------
    bl = direct["by_prompt_len"]
    Ps = [int(k) for k in bl]
    vs = [bl[k]["value"] for k in bl]
    es = [0.0 if math.isnan(bl[k]["stderr"]) else bl[k]["stderr"] for k in bl]
    ax[0, 0].errorbar(Ps, vs, yerr=es, marker="o", ms=7, lw=2,
                      color=C_PREFILL, capsize=4)
    ax[0, 0].set_xscale("log", base=2)
    ax[0, 0].set_xticks(Ps); ax[0, 0].set_xticklabels([str(p) for p in Ps])
    ax[0, 0].set_ylim(0, max(vs) * 1.4)
    ax[0, 0].set_xlabel("prompt length (tokens)")
    ax[0, 0].set_ylabel("joules per prefill token")
    ax[0, 0].set_title("prefill cost, measured directly", loc="left")

    # -- (0,1) THE MONEY PLOT: decode cost vs batch ----------------------
    if sweep:
        bs = [int(k) for k in sweep]
        dv = [sweep[k]["value"] for k in sweep]
        de = [0.0 if math.isnan(sweep[k]["stderr"]) else sweep[k]["stderr"]
              for k in sweep]
        ax[0, 1].errorbar(bs, dv, yerr=de, marker="o", ms=7, lw=2,
                          color=C_DECODE, capsize=4, label="measured",
                          zorder=3)
        # Ideal 1/batch reference, anchored at the smallest batch. Departure
        # from this line is the point where batching stops paying.
        ideal = [dv[0] * bs[0] / x for x in bs]
        ax[0, 1].plot(bs, ideal, ls="--", lw=1.2, color=C_MUTED,
                      label="ideal 1/batch", zorder=2)
        ax[0, 1].set_xscale("log", base=2); ax[0, 1].set_yscale("log")
        ax[0, 1].set_xticks(bs); ax[0, 1].set_xticklabels([str(x) for x in bs])
        ax[0, 1].legend(loc="upper right")
    ax[0, 1].set_xlabel("batch size")
    ax[0, 1].set_ylabel("joules per decode token")
    ax[0, 1].set_title("decode cost vs batch  (the scheduler's lever)",
                       loc="left")

    # -- (0,2) power by workload mix -------------------------------------
    grid = [r for r in runs if r.phase == "grid"]
    ax[0, 2].scatter([r.decode_share for r in grid], [r.mean_w for r in grid],
                     s=28, color=C_PREFILL, edgecolor="white", linewidth=0.6,
                     zorder=3)
    ax[0, 2].axhline(idle_w, ls=":", lw=1.3, color=C_MUTED)
    ax[0, 2].annotate(f"parked {idle_w:.0f} W", (1.0, idle_w),
                      textcoords="offset points", xytext=(-2, 6), ha="right",
                      fontsize=8, color=C_MUTED)
    if static.get("power_limit_w"):
        ax[0, 2].axhline(static["power_limit_w"], ls="--", lw=1.3,
                         color=C_CAPPED)
        ax[0, 2].annotate(f"cap {static['power_limit_w']:.0f} W",
                          (1.0, static["power_limit_w"]),
                          textcoords="offset points", xytext=(-2, 5),
                          ha="right", fontsize=8, color=C_CAPPED)
    ax[0, 2].set_ylim(0, (static.get("power_limit_w") or 300) * 1.12)
    ax[0, 2].set_xlabel("decode share of tokens")
    ax[0, 2].set_ylabel("mean power (W)")
    ax[0, 2].set_title("the gap that matters: busy vs parked", loc="left")

    # -- (1,0) does the cap bind? -----------------------------------------
    # y is the driver's own answer: the fraction of samples in the window
    # where the SW_POWER_CAP throttle bit was set. A window can average well
    # under the limit and still be capped most of the time, because the cap
    # acts on instantaneous peaks (260-350 W here) and clocks down to bring
    # them back. So "mean power < limit" is not headroom; this flag is.
    capped = [r for r in runs if r.frac_power_capped > 0.05]
    clean = [r for r in runs if r.frac_power_capped <= 0.05]
    for grp, col, lab in ((clean, C_OK, "cap flag < 5 % of samples"),
                          (capped, C_CAPPED, "cap flag > 5 %")):
        if grp:
            ax[1, 0].scatter([r.mean_w for r in grp],
                             [100 * r.frac_power_capped for r in grp], s=30,
                             color=col, label=lab, edgecolor="white",
                             linewidth=0.6)
    if static.get("power_limit_w"):
        ax[1, 0].axvline(static["power_limit_w"], ls="--", lw=1.2,
                         color=C_CAPPED)
        ax[1, 0].annotate(f"limit {static['power_limit_w']:.0f} W",
                          (static["power_limit_w"], 50),
                          textcoords="offset points", xytext=(-4, 0),
                          ha="right", fontsize=8, color=C_CAPPED)
    ax[1, 0].set_ylim(-3, 103)
    ax[1, 0].set_xlabel("mean power over window (W)")
    ax[1, 0].set_ylabel("% of samples with SW power-cap flag set")
    ax[1, 0].set_title("does the cap bind?  (driver's own flag)", loc="left")
    ax[1, 0].legend(loc="upper left")

    # -- (1,1) cross-check validity --------------------------------------
    pred = np.array([cross["a"] * r.prefill_tokens + cross["b"] * r.decode_tokens
                     + cross["d"] * r.inner_reps
                     + cross["fitted_baseline_w"] * r.wall_s for r in runs])
    meas = np.array([r.energy_j for r in runs])
    lim = [0, max(pred.max(), meas.max()) * 1.08]
    ax[1, 1].plot(lim, lim, ls="--", lw=1, color=C_MUTED)
    ok = cross["trustworthy"]
    ax[1, 1].scatter(pred, meas, s=26, color=C_OK if ok else C_CAPPED,
                     alpha=0.85, edgecolor="white", linewidth=0.6)
    ax[1, 1].set_xlim(lim); ax[1, 1].set_ylim(lim)
    ax[1, 1].set_xlabel("predicted total energy (J)")
    ax[1, 1].set_ylabel("measured total energy (J)")
    ax[1, 1].set_title(
        f"cross-check: baseline fitted {cross['fitted_baseline_w']:.0f} W "
        f"vs measured {cross['measured_parked_w']:.0f} W  "
        f"[{'PASS' if ok else 'FAIL'}]", loc="left",
        color=C_INK if ok else C_CAPPED)

    # -- (1,2) power traces ----------------------------------------------
    def trace_for(group):
        if not group:
            return None, None
        tgt = max(group, key=lambda r: r.mean_w)
        for k, v in traces.items():
            if (f"_P{tgt.prompt_tokens}_" in k and f"_D{tgt.output_tokens}_" in k
                    and k.endswith(f"_b{tgt.batch}_rep{tgt.rep}")):
                return v, tgt
        return None, None

    for grp, col, lab in ((capped, C_CAPPED, "power-capped window"),
                          (clean, C_OK, "full-clock window")):
        tr, tgt = trace_for(grp)
        if tr and tr["t"]:
            # The legend carries identity here; per-line annotations collided
            # with the traces, so the config goes in the legend label instead.
            ax[1, 2].plot(tr["t"], tr["w"], lw=1.8, color=col,
                          label=f"{lab}  (P={tgt.prompt_tokens} "
                                f"D={tgt.output_tokens})")
    ax[1, 2].axhline(idle_w, ls=":", lw=1.2, color=C_MUTED)
    if static.get("power_limit_w"):
        ax[1, 2].axhline(static["power_limit_w"], ls="--", lw=1.2,
                         color=C_MUTED)
    ax[1, 2].set_ylim(0, (static.get("power_limit_w") or 300) * 1.12)
    ax[1, 2].set_xlabel("seconds into window")
    ax[1, 2].set_ylabel("power (W)")
    ax[1, 2].set_title("power trace", loc="left")
    ax[1, 2].legend(loc="lower right")

    fig.suptitle(f"Jade {PROBE_VERSION} -- {args.model}   "
                 f"({args.reps} reps, >= {args.min_seconds}s windows, "
                 f"cooled to {args.cool_to:.0f} C)", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(f"{args.out_prefix}.png", bbox_inches="tight")
    print(f"[jade] wrote {args.out_prefix}.png", flush=True)


if __name__ == "__main__":
    main()
