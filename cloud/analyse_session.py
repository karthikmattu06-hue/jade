#!/usr/bin/env python3
"""
analyse_session.py -- one summary row per event from a lambda_session.sh run.

Complements jade_e3_rows.py (schema rows) with the numbers the 11 Nov write-up
needed and the rows script does not print:
  * mean power per pool, baseline (last 30 s before the command) vs hold (second half)
  * mean SM clock in the hold (did a cap actually slow the clock? did the lock hold?)
  * TTFT and TPOT p50/p99 for ALL served tiers, baseline vs hold
    (the rows script reports TTFT by tier; a decode lever mostly costs TPOT)
  * release: peak 1 s-mean power after the release vs the peak in the baseline
    (schema rev 3 proposal: compare against the baseline's own peak, because
    prefill bursts normally reach 250-340 W) and premium p99 TTFT after release

Usage: python analyse_session.py ~/jade_runs/session_<date>   -> events.csv in the cwd
"""
import csv, sys, os, statistics as st, math, bisect
S = sys.argv[1]
def pct(xs, q):
    xs = sorted(x for x in xs if x is not None)
    return xs[min(len(xs)-1, max(0, math.ceil(q/100*len(xs))-1))] if xs else float('nan')
def load_power(p):
    d = {}
    for r in csv.DictReader(open(p)):
        d.setdefault(r['role'], []).append((float(r['t_wall']), float(r['power_w']), float(r['sm_mhz'])))
    return d
def win(series, a, b, i=1):
    ts = [x[0] for x in series]; lo, hi = bisect.bisect_left(ts, a), bisect.bisect_left(ts, b)
    return [x[i] for x in series[lo:hi]]
out = []
for k in range(4):
  for step in ('levers', 'admission_ev'):
    d = f'{S}/pair{k}/{step}'
    if not os.path.exists(f'{d}/e3_events.csv'): continue
    pw = load_power(f'{S}/pair{k}/power.csv')
    rq = list(csv.DictReader(open(f'{d}/e3_requests.csv')))
    for ev in csv.DictReader(open(f'{d}/e3_events.csv')):
        t0, tc, tr, te = (float(ev[x]) for x in ('t_start','t_cmd','t_rel','t_end'))
        hold = tr - tc
        er = [r for r in rq if r['event'] == ev['event']]
        def lat(a, b, f, tiers=('premium','standard','flex')):
            return [float(r[f]) if r[f] and r['status']=='ok' else (math.inf if r['status'] not in ('ok','http429') else None)
                    for r in er if r['tier'] in tiers and a <= float(r['t_sched']) < b and r['status'] != 'http429']
        row = dict(pair=k, step=step, ev=ev['event'], kind=ev['kind'], rate=float(ev['rate']))
        tot_b = tot_h = 0
        for pool in ('prefill','decode'):
            s = pw[pool]
            b = st.mean(win(s, max(t0, tc-30), tc)); h = st.mean(win(s, tc+hold/2, tr))
            row[f'{pool}_base'] = b; row[f'{pool}_hold'] = h
            row[f'{pool}_mhz_hold'] = st.mean(win(s, tc+hold/2, tr, 2))
            # release: peak 1 s-mean power in 60 s after release vs peak in the baseline window
            def peak(a, bb):
                xs = win(s, a, bb); return max(st.mean(xs[i:i+20]) for i in range(0, max(1, len(xs)-20), 5)) if len(xs) > 20 else float('nan')
            row[f'{pool}_peak_base'] = peak(t0, tc); row[f'{pool}_peak_after'] = peak(tr, te)
            tot_b += b; tot_h += h
        row['shed_w'] = tot_b - tot_h
        for f in ('ttft_ms', 'tpot_ms'):
            for q in (50, 99):
                row[f'{f}_p{q}_base'] = pct(lat(t0, tc, f), q); row[f'{f}_p{q}_hold'] = pct(lat(tc, tr, f), q)
        row['prem_ttft_p99_after'] = pct(lat(tr, te, 'ttft_ms', ('premium',)), 99)
        row['prem_ttft_p99_base'] = pct(lat(t0, tc, 'ttft_ms', ('premium',)), 99)
        row['n_hold'] = len(lat(tc, tr, 'ttft_ms'))
        row['ok_frac_all'] = sum(r['status']=='ok' for r in er)/max(1,sum(r['status']!='http429' for r in er))
        out.append(row)
w = csv.DictWriter(open('events.csv','w',newline=''), fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
print(len(out), 'events')
