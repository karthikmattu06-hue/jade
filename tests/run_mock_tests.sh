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
#  Runtime ~1 min. Needs python3 with aiohttp (pip install aiohttp).
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
echo "outputs in $OUT"; exit $FAIL
