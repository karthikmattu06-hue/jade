#!/bin/bash
# ======================================================================
#  run_mock_tests.sh -- end-to-end test of Jade's tools on a laptop, no GPU.
#
#  mock_vllm.py (fake server) <- jade_admission_proxy.py (the lever) <- jade_loadgen.py / jade_e3.py
#  then jade_e3_rows.py on synthetic power with known answers.
#  Checks (each prints PASS/FAIL; exit code = number of failures):
#    1. open-loop load generator completes every request at low load
#    2. admission proxy refuses flex (429) during a reject shed, holds-and-serves flex during defer
#    3. every lever command is logged with a timestamp
#    4. datasheet rows recover known depth (drain 0.64, shed 0.08) and rebound (defer 0.12)
#    5. ramped release spreads the deferred backlog over the ramp; instant release does not
#    6. root events (cap / clock) call nvidia-smi with the right arguments on the right GPU,
#       restore defaults at release and at the end, and get schema lever names in the rows
#       (uses fake `sudo` and `nvidia-smi` scripts on PATH -- nothing touches a real GPU)
#  Runtime ~2 min. Needs python3 with aiohttp (pip install aiohttp).
# ======================================================================
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"; SRC="$HERE/../disagg"
OUT="$(mktemp -d)"; FAIL=0
pass() { echo "PASS  $1"; }; fail() { echo "FAIL  $1"; FAIL=$((FAIL+1)); }

python3 "$HERE/mock_vllm.py" --port 18999 & MOCK=$!
python3 "$SRC/jade_admission_proxy.py" --port 18300 --upstream localhost:18999 --log-dir "$OUT" > "$OUT/proxy.log" 2>&1 & PX=$!
trap 'kill $MOCK $PX 2>/dev/null' EXIT
sleep 2

# tiny synthetic trace: (prompt, output) lengths
python3 - "$OUT/trace.csv" <<'PY'
import csv, random, sys
r = random.Random(1); w = csv.writer(open(sys.argv[1], "w", newline=""))
w.writerow(["TIMESTAMP", "ContextTokens", "GeneratedTokens"])
for i in range(300): w.writerow([i, r.randint(50, 2000), r.randint(2, 60)])
PY

# 1. load generator, straight at the mock
python3 "$SRC/jade_loadgen.py" --url localhost:18999 --model mock --trace "$OUT/trace.csv" \
  --rates 2 4 --dur-s 5 --tail-s 3 --gap-s 0.5 --idle-s 0.5 --out-dir "$OUT/lg" > "$OUT/lg.txt" 2>&1
python3 - "$OUT/lg/e1_requests.csv" <<'PY' && pass "loadgen: all requests completed" || fail "loadgen"
import csv, sys; rows = list(csv.DictReader(open(sys.argv[1])))
sys.exit(0 if rows and all(r["status"] == "ok" for r in rows) else 1)
PY

# 2+3. E3 runner through the admission proxy (short windows)
python3 "$SRC/jade_e3.py" --proxy localhost:18300 --model mock --trace "$OUT/trace.csv" --rates 4 \
  --base-s 3 --hold-s 4 --after-s 3 --tail-s 2 --gap-s 0.5 --idle-s 0.5 --out-dir "$OUT" > "$OUT/e3.txt" 2>&1
python3 - "$OUT" <<'PY' && pass "admission: reject refuses flex, defer serves it, others untouched" || fail "admission behaviour"
import csv, sys, os
ev = {r["event"]: r for r in csv.DictReader(open(os.path.join(sys.argv[1], "e3_events.csv")))}
rq = list(csv.DictReader(open(os.path.join(sys.argv[1], "e3_requests.csv"))))
def hold(kind, tier):
    e = next(e for e in ev.values() if e["kind"] == kind)
    return [r for r in rq if r["event"] == e["event"] and r["tier"] == tier and float(e["t_cmd"]) + 0.1 <= float(r["t_sched"]) < float(e["t_rel"])]
ok = all(r["status"] == "http429" for r in hold("shed_reject", "flex")) \
     and all(r["status"] == "ok" for r in hold("shed_defer", "flex")) \
     and all(r["status"] == "ok" for r in hold("shed_reject", "premium"))
sys.exit(0 if ok and hold("shed_reject", "flex") else 1)
PY
[ "$(grep -c release "$OUT/commands.csv")" -ge 3 ] && pass "proxy: every command logged" || fail "command log"

# 4. datasheet rows on synthetic power with known answers
python3 "$HERE/synthetic_power.py" "$OUT"
python3 "$SRC/jade_e3_rows.py" --dir "$OUT" > "$OUT/rows.txt" 2>&1
python3 - "$OUT/e3_rows.csv" <<'PY' && pass "rows: depth and rebound match the known answers" || fail "rows"
import csv, sys
r = {(x["kind"], x["pool"]): x for x in csv.DictReader(open(sys.argv[1]))}
ok = abs(float(r[("drain", "decode")]["depth_frac"]) - 0.64) < 0.03 \
     and abs(float(r[("shed_reject", "decode")]["depth_frac"]) - 0.08) < 0.02 \
     and abs(float(r[("shed_defer", "decode")]["rebound_frac"]) - 0.12) < 0.02 \
     and float(r[("drain", "decode")]["speed_s"]) < 1.0
sys.exit(0 if ok else 1)
PY
# 5. ramped release (added 11 Nov): defer vs defer_ramp, release spread of the backlog
mkdir -p "$OUT/ramp"
python3 "$SRC/jade_admission_proxy.py" --port 18301 --upstream localhost:18999 --log-dir "$OUT/ramp" > "$OUT/ramp/proxy.log" 2>&1 & PX2=$!
trap 'kill $MOCK $PX $PX2 2>/dev/null' EXIT
sleep 1.5
python3 "$SRC/jade_e3.py" --proxy localhost:18301 --model mock --trace "$OUT/trace.csv" --rates 6 \
  --events shed_defer shed_defer_ramp --ramp-s 3 \
  --base-s 2 --hold-s 4 --after-s 5 --tail-s 2 --gap-s 0.5 --idle-s 0.5 --out-dir "$OUT/ramp" > "$OUT/ramp/e3.txt" 2>&1
python3 - "$OUT/ramp" <<'PY' && pass "ramp: backlog released over the ramp, instant release at once" || fail "ramped release"
import csv, sys, os
d = sys.argv[1]
ev = list(csv.DictReader(open(os.path.join(d, "e3_events.csv"))))
adm = [r for r in csv.DictReader(open(os.path.join(d, "admissions.csv"))) if r["decision"] == "deferred"]
def spread(e):     # seconds between the first and last release of this event's backlog
    rel = [float(r["t_in"]) + float(r["hold_ms"]) / 1000 for r in adm
           if float(e["t_cmd"]) <= float(r["t_in"]) < float(e["t_rel"])]
    return (max(rel) - min(rel)) if len(rel) >= 3 else None, len(rel)
(s_i, n_i), (s_r, n_r) = spread(ev[0]), spread(ev[1])
print(f"      instant: {n_i} held, released over {s_i:.2f} s | ramp 3 s: {n_r} held, released over {s_r:.2f} s")
sys.exit(0 if s_i is not None and s_r is not None and s_i < 0.3 and 1.8 < s_r < 3.5 else 1)
PY

# 6. root levers with fake sudo / nvidia-smi
FAKE="$OUT/fakebin"; mkdir -p "$FAKE"
cat > "$FAKE/sudo" <<'SH'
#!/bin/bash
[ "$1" = "-n" ] && shift
exec "$@"
SH
cat > "$FAKE/nvidia-smi" <<SH
#!/bin/bash
echo "\$*" >> "$OUT/smi_calls.txt"
case "\$*" in *power.default_limit*) echo "400.00" ;; *) echo "OK" ;; esac
SH
chmod +x "$FAKE/sudo" "$FAKE/nvidia-smi"
mkdir -p "$OUT/root"
PATH="$FAKE:$PATH" python3 "$SRC/jade_e3.py" --proxy localhost:18301 --model mock --trace "$OUT/trace.csv" --rates 2 \
  --events cap:decode:225 clock:prefill:810 --prefill-gpu 4 --decode-gpu 5 \
  --base-s 1 --hold-s 2 --after-s 1 --tail-s 1 --gap-s 0.3 --idle-s 0.3 --out-dir "$OUT/root" > "$OUT/root/e3.txt" 2>&1
python3 - "$OUT" <<'PY' && pass "root: right commands on the right GPUs, defaults restored" || fail "root levers"
import sys, os
calls = open(os.path.join(sys.argv[1], "smi_calls.txt")).read().splitlines()
calls = [c for c in calls if "--query-gpu" not in c]
want_in_order = ["-i 5 -pl 225", "-i 5 -pl 400", "-i 4 -lgc 810,810", "-i 4 -rgc"]
# each wanted call must appear AFTER the previous one (the initial reset also sends "-pl 400")
i, ok = -1, True
for w in want_in_order:
    i = next((j for j in range(i + 1, len(calls)) if calls[j] == w), None)
    if i is None:
        ok = False; break
ends_reset = calls[-4:] == ["-i 4 -pl 400", "-i 4 -rgc", "-i 5 -pl 400", "-i 5 -rgc"]
print("      calls:", " ; ".join(calls))
sys.exit(0 if ok and ends_reset else 1)
PY
cp "$OUT/e3_power.csv" "$OUT/root/e3_power.csv"      # any power file will do: this checks naming, not numbers
python3 - "$OUT/root" <<'PY'
# shift the synthetic power timestamps onto this run's events so the windows are not empty
import csv, sys, os
d = sys.argv[1]
ev = list(csv.DictReader(open(os.path.join(d, "e3_events.csv"))))
t0, t1 = float(ev[0]["t_start"]) - 5, float(ev[-1]["t_end"]) + 5
rows = list(csv.DictReader(open(os.path.join(d, "e3_power.csv"))))
w = csv.DictWriter(open(os.path.join(d, "e3_power.csv"), "w", newline=""), fieldnames=list(rows[0])); w.writeheader()
t, i = t0, 0
while t < t1:
    for r in rows[i % len(rows): i % len(rows) + 2]:
        r = dict(r); r["t_wall"] = f"{t:.4f}"; w.writerow(r)
    t += 0.05; i += 2
PY
python3 "$SRC/jade_e3_rows.py" --dir "$OUT/root" --nominal-w 400 > "$OUT/root/rows.txt" 2>&1
python3 - "$OUT/root/e3_rows.json" <<'PY' && pass "root rows: schema lever names and settings" || fail "root rows"
import json, sys
j = json.load(open(sys.argv[1]))
names = {(x["name"], x["pool"], json.dumps(x["setting"], sort_keys=True)) for x in j}
ok = ("power_cap", "decode", json.dumps({"gpu_pool": "decode", "cap_w": 225}, sort_keys=True)) in names \
     and ("sm_clock_lock", "prefill", json.dumps({"gpu_pool": "prefill", "sm_mhz": 810}, sort_keys=True)) in names
sys.exit(0 if ok else 1)
PY

echo "outputs in $OUT"; exit $FAIL
