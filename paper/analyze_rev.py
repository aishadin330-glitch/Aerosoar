"""Analysis of the revision experiments -> paper/results/rev_summary.json (printed compactly)."""
import json, numpy as np, sys
from collections import defaultdict
from scipy import signal as sg

import pathlib
R = str(pathlib.Path(__file__).resolve().parent / 'results') + '/'
L = lambda n: json.load(open(R + n))
rng = np.random.default_rng(1)
out = {}

def ci(x, f=np.mean, n=2000):
    x = np.asarray(x, float)
    b = [f(x[rng.integers(0, len(x), len(x))]) for _ in range(n)]
    return float(f(x)), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))

# ---------------------------------------------------------------- latency
lat = {}
for part in ('fc', 'gnss', 'pitot'):
    g = defaultdict(list)
    for r in L(f'rev_latency_{part}.json'): g[r['name']].append(r)
    for k, v in g.items():
        a = [r for r in v if r['arrived']]
        lat[k] = dict(n=len(v), arr=len(a), speed=np.mean([r['speed'] for r in a]) if a else None,
                      wh=np.mean([r['wh'] for r in a]) if a else None, used=np.mean([r['used'] for r in a]) if a else None,
                      rej=np.mean([r['rejected'] for r in a]) if a else None,
                      acc=np.nanmean([r['acc'] for r in a if r['acc'] is not None]) if any(r['acc'] is not None for r in a) else None,
                      ctr=np.nanmean([r['ctr'] for r in a if r['ctr'] is not None]) if any(r['ctr'] is not None for r in a) else None)
out['latency'] = lat

# ---------------------------------------------------------------- ROC
roc = L('rev_roc.json')
keys = list(roc[0]['det'].keys())
tab = {}
for turb in (0.2, 0.5, 0.9):
    for split in ('tune', 'test'):
        rs = [r for r in roc if r['split'] == split and r['turb'] == turb]
        enc = sum(r['enc'] for r in rs); T = sum(r['t'] for r in rs) / 3600
        for k in keys:
            hit = sum(r['det'][k]['hit'] for r in rs); fa = sum(r['det'][k]['fa'] for r in rs)
            lats = [x for r in rs for x in r['det'][k]['lat']]
            tab[f'{turb}|{split}|{k}'] = dict(pd=hit / enc, fa=fa / T, lat=float(np.mean(lats)) if lats else None, enc=enc, hours=T)
out['roc'] = tab
# operating points: choose on tune seeds the setting with fa <= 5/h (best Pd), report on test with bootstrap over flights
sel = {}
for turb in (0.2, 0.5, 0.9):
    for fam in ('thr', 'fix', 'ada'):
        cands = [k for k in keys if k.startswith(fam)]
        best = None
        for k in cands:
            t = tab[f'{turb}|tune|{k}']
            if t['fa'] <= 5 and (best is None or t['pd'] > tab[f'{turb}|tune|{best}']['pd']): best = k
        sel[f'{turb}|{fam}'] = best
out['roc_sel'] = sel
boot = {}
for turb in (0.2, 0.5, 0.9):
    rs = [r for r in roc if r['split'] == 'test' and r['turb'] == turb]
    for fam in ('thr', 'fix', 'ada'):
        k = sel[f'{turb}|{fam}']
        if k is None: continue
        enc = np.array([r['enc'] for r in rs]); hit = np.array([r['det'][k]['hit'] for r in rs]); fa = np.array([r['det'][k]['fa'] for r in rs]); T = np.array([r['t'] for r in rs]) / 3600
        lt = np.array([np.mean(r['det'][k]['lat']) if r['det'][k]['lat'] else np.nan for r in rs])
        P, F = [], []
        for _ in range(2000):
            i = rng.integers(0, len(rs), len(rs))
            P.append(hit[i].sum() / enc[i].sum()); F.append(fa[i].sum() / T[i].sum())
        boot[f'{turb}|{fam}'] = dict(k=k, pd=[hit.sum() / enc.sum(), *np.percentile(P, [2.5, 97.5])], fa=[fa.sum() / T.sum(), *np.percentile(F, [2.5, 97.5])], lat=float(np.nanmean(lt)))
out['roc_boot'] = boot
# ARL0 estimate: mean time between false alarms = 3600/fa-rate (flight-averaged)
# ---------------------------------------------------------------- Kalman
kal = defaultdict(list)
for part in ('on', 'off'):
    for r in L(f'rev_kalman_{part}.json'): kal[(r['gnss'], r['baro'], r['qb'], r['qj'])].append(r)
kt = {}
for k, v in kal.items():
    pool = lambda a, n: float(np.sqrt(sum((r[a] or 0) ** 2 * r[n] for r in v) / max(1, sum(r[n] for r in v))))
    kt[str(k)] = dict(steady=pool('steady', 'ns'), manoeuvre=pool('manoeuvre', 'nm'), ns=sum(r['ns'] for r in v), nm=sum(r['nm'] for r in v))
out['kalman'] = kt
# ---------------------------------------------------------------- stress
st = defaultdict(list)
for r in L('rev_stress.json'): st[r['cond']].append(r)
stt = {}
for c, v in st.items():
    d = dict(A=float(np.sqrt(np.mean([r['rmse']['A'] ** 2 for r in v]))), B=float(np.sqrt(np.mean([r['rmse']['B'] ** 2 for r in v]))), C=float(np.sqrt(np.mean([r['rmse']['C'] ** 2 for r in v]))),
             biasB=float(np.mean([r['bias']['B'] for r in v])), biasC=float(np.mean([r['bias']['C'] for r in v])))
    T = sum(r['det']['t'] for r in v) / 3600
    enc = sum(r['det']['enc'] for r in v)
    for k in ('thr|1', 'fix|1.5', 'ada|1.5'):
        hit = sum(r['det']['det'][k]['hit'] for r in v); fa = sum(r['det']['det'][k]['fa'] for r in v)
        lats = [x for r in v for x in r['det']['det'][k]['lat']]
        d[k] = dict(fa=fa / T, pd=hit / enc, lat=float(np.mean(lats)) if lats else None)
    d['enc'] = enc
    stt[c] = d
out['stress'] = stt
# ---------------------------------------------------------------- offline filter comparison on pipeline-B signals
sig = L('rev_signals.json')
fs = 20.0
def causal_sg(N, p):  # least-squares polynomial derivative at the newest sample, applied to the energy height
    c = sg.savgol_coeffs(N, p, deriv=1, delta=1 / fs, pos=N - 1)
    f = lambda z: sg.lfilter(c, [1.0], z)
    f.on_eh = True
    return f
def butter(fc):
    b, a = sg.butter(2, fc / (fs / 2)); return lambda z: sg.lfilter(b, a, z)
def bessel(fc):
    b, a = sg.bessel(2, fc / (fs / 2), norm='mag'); return lambda z: sg.lfilter(b, a, z)
def ma(fc):
    N = int(round(0.443 * fs / fc)); return lambda z: sg.lfilter(np.ones(N) / N, [1.0], z)
def onepole(fc):
    al = 1 - np.exp(-2 * np.pi * fc / fs); return lambda z: sg.lfilter([al], [1, al - 1], z)
FILT = {'Butterworth (2nd order)': butter(0.4), 'Bessel (2nd order)': bessel(0.4), 'Moving average': ma(0.4), 'One-pole': onepole(0.4),
        'Causal Savitzky-Golay (N=41, p=2)': causal_sg(41, 2), 'Causal Savitzky-Golay (N=61, p=2)': causal_sg(61, 2), 'Causal Savitzky-Golay (N=81, p=2)': causal_sg(81, 2)}
def evalfilt(fn):
    se = 0; n = 0; best = {}
    for r in sig:
        w = np.array(r['w']); y = fn(np.array(r['eh'] if getattr(fn, 'on_eh', False) else r['ed'])) + np.array(r['sink'])
        for sh in range(0, 121, 2):  # shift truth by sh samples to expose the delay-free error
            e = y[300 + sh:] - w[300:len(w) - sh] if sh else y[300:] - w[300:]
            best.setdefault(sh, [0, 0]); best[sh][0] += float(np.sum(e ** 2)); best[sh][1] += len(e)
    c0 = np.sqrt(best[0][0] / best[0][1])
    shs = {k: np.sqrt(v[0] / v[1]) for k, v in best.items()}
    k = min(shs, key=shs.get)
    return float(c0), float(shs[k]), k / fs
sg_tab = {}
for name, fn in FILT.items():
    c0, cb, d = evalfilt(fn)
    sg_tab[name] = dict(rmse=c0, rmse_aligned=cb, delay=d)
# delay of each filter measured from the impulse centroid
def gd(fn):
    x = np.zeros(2000); x[0] = 1; y = fn(np.cumsum(x) / fs if getattr(fn, 'on_eh', False) else x); return float(np.sum(np.arange(2000) * y) / np.sum(y) / fs)
for name, fn in FILT.items(): sg_tab[name]['gd'] = gd(fn)
out['filters_offline'] = sg_tab
# observability of the vertical channel (state [h, vz, accel bias, baro offset])
dt = 0.01
F = np.array([[1, dt, -0.5 * dt * dt, 0], [0, 1, -dt, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
def rank(Hs):
    O = np.vstack([h @ np.linalg.matrix_power(F, k) for h in Hs for k in range(4)]) if False else None
    rows = []
    for k in range(40):
        Fk = np.linalg.matrix_power(F, k)
        for h in Hs: rows.append(np.array(h, float) @ Fk)
    O = np.array(rows); sv = np.linalg.svd(O, compute_uv=False)
    return int(np.sum(sv > 1e-9 * sv[0])), [float(x) for x in sv]
out['obs'] = {'baro only': rank([[1, 0, 0, 1]])[0], 'baro + GNSS altitude': rank([[1, 0, 0, 1], [1, 0, 0, 0]])[0]}
json.dump(out, open(R + 'rev_summary.json', 'w'))

def p(*a): print(*a)
p('FILTERS',json.dumps(out['filters_offline'],indent=0));p('OBS',out['obs']);p('LATENCY'); [p(k, {a: (round(b, 3) if isinstance(b, float) else b) for a, b in v.items()}) for k, v in lat.items()]
p('ROC selected'); p(sel)
for k, v in boot.items(): p(k, v['k'], 'pd %.3f [%.3f %.3f]' % tuple(v['pd']), 'fa %.2f [%.2f %.2f]' % tuple(v['fa']), 'lat %.2f' % v['lat'])
p('KALMAN')
for k, v in kt.items(): p(k, round(v['steady'], 3), round(v['manoeuvre'], 3))
p('STRESS')
for c, d in stt.items(): p(c, {a: (round(b, 3) if isinstance(b, float) else b) for a, b in d.items() if a not in ('enc',)})
