"""Gauntlet: MODEL-PROBABILITY meta-edge (genuinely UNEXPLORED).

The deepest unasked question: does the pipeline's OWN ML prediction (model_p_5d)
have any realizable, net-of-cost, OOS, day-demeaned edge? If the whole machine
has any signal, its confidence should cross-sectionally sort forward 5d returns.

Hypotheses:
  M1) Cross-sectional rank of model_p_5d within day -> long high-conf, short low-conf.
  M2) Threshold buckets: long p>=0.6, short p<=0.4 (the model's own buy/sell calls).
  M3) Borrowable subset (px>=10) so the short side is realizable.
  M4) Conditioned on model agreeing with high source_weight / factual (clean signals).

Tradeable framing: model_p_5d is known at scored_at (~published_at for live-timed
rows). Forward return runs from price_at_flag. We restrict to LIVE-TIMED rows
(|scored_at - published_at| <= 2d) to kill backfill artifact.

Gauntlet: DEDUP (one bet/ticker-day), DAY-DEMEAN, NET cost (price-bucketed; +borrow
for shorts 0.2%/day), OOS 70/30 by day, DROP-TOP-5, PERMUTATION p<0.05, >=15 days.
Bonferroni context: ~240+ prior tests -> a lone p<0.05 is NOISE.
"""
import sqlite3, numpy as np
from collections import defaultdict

DB = "data/market_radar.db"

def rt_cost(px):
    if px <= 0: return 0.040
    if px < 1: return 0.040
    if px < 3: return 0.030
    if px < 5: return 0.020
    if px < 10: return 0.012
    if px < 50: return 0.005
    return 0.002

con = sqlite3.connect(DB)
con.row_factory = sqlite3.Row
rows = con.execute("""
  SELECT ss.ticker AS ticker, substr(rs.published_at,1,10) AS day,
         so.return_1d_pct AS r1, so.return_5d_pct AS r5,
         so.price_at_flag AS px, ss.model_p_5d AS p,
         ss.source_weight AS sw, ss.factual AS fac, ss.sentiment AS sent
  FROM signal_scores ss
  JOIN raw_signals rs ON rs.id=ss.signal_id
  JOIN signal_outcomes so ON so.score_id=ss.id
  WHERE so.return_5d_pct IS NOT NULL
    AND COALESCE(so.data_corrupt,0)=0
    AND so.price_at_flag BETWEEN 1 AND 2000
    AND ABS(so.return_5d_pct)<=100
    AND ss.model_p_5d IS NOT NULL
    AND rs.published_at IS NOT NULL
    AND ABS(julianday(ss.scored_at)-julianday(rs.published_at))<=2
""").fetchall()
print(f"raw live-timed clean rows={len(rows)}")

# DEDUP: one bet per (ticker, day) — average the duplicates (model_p, returns, px)
panel = defaultdict(lambda: {"n":0,"r1":0.0,"r5":0.0,"px":0.0,"p":0.0,"sw":0.0,"fac":0.0,"sent":0.0})
for r in rows:
    k=(r["ticker"], r["day"]); pp=panel[k]; pp["n"]+=1
    pp["r1"]+= r["r1"] if r["r1"] is not None else 0.0
    pp["r5"]+= r["r5"]
    pp["px"]+= r["px"]
    pp["p"] += r["p"]
    pp["sw"]+= r["sw"] if r["sw"] is not None else 0.0
    pp["fac"]+= r["fac"] if r["fac"] is not None else 0.0
    pp["sent"]+= r["sent"] if r["sent"] is not None else 0.0
recs=[]
for (tk,day),pp in panel.items():
    n=pp["n"]
    recs.append({"ticker":tk,"day":day,"r1":pp["r1"]/n,"r5":pp["r5"]/n,
                 "px":pp["px"]/n,"p":pp["p"]/n,"sw":pp["sw"]/n,
                 "fac":pp["fac"]/n,"sent":pp["sent"]/n})
print(f"deduped ticker-days={len(recs)}")

byday=defaultdict(list)
for r in recs: byday[r["day"]].append(r)
dense_days=sorted(d for d,v in byday.items() if len(v)>=20)
print(f"dense days (>=20 tk)={len(dense_days)}: {dense_days}")

def daily(horizon, top_frac, side, min_px=1.0, min_sw=None, require_fac=False, net=True):
    """Day-demeaned NET basket return. side in {long_high,short_low,long_high_short_low,
    short_high,long_low}."""
    out=[]
    for d in dense_days:
        v=[r for r in byday[d] if r["px"]>=min_px]
        if min_sw is not None: v=[r for r in v if r["sw"]>=min_sw]
        if require_fac: v=[r for r in v if r["fac"]>=0.5]
        if len(v)<20: continue
        v=sorted(v,key=lambda r:r["p"])
        k=max(1,int(len(v)*top_frac))
        daymean=np.mean([r[horizon] for r in v])
        hi=v[-k:]; lo=v[:k]
        bdays = 5 if horizon=="r5" else 1
        def leg(group, short):
            vals=[]
            for r in group:
                g=r[horizon]
                if short: g=-g
                if net:
                    g-= rt_cost(r["px"])*100
                    if short: g-= 0.2*bdays
                dm = daymean if not short else -daymean
                vals.append(g-dm)
            return np.mean(vals)
        if side=="long_high": out.append((d, leg(hi,False)))
        elif side=="short_low": out.append((d, leg(lo,True)))
        elif side=="short_high": out.append((d, leg(hi,True)))
        elif side=="long_low": out.append((d, leg(lo,False)))
        elif side=="long_high_short_low":
            out.append((d, 0.5*leg(hi,False)+0.5*leg(lo,True)))
    return out

def thresh(horizon, side, min_px=1.0, net=True):
    """Model's own buy/sell calls: long p>=0.6, short p<=0.4."""
    out=[]
    for d in dense_days:
        v=[r for r in byday[d] if r["px"]>=min_px]
        if len(v)<20: continue
        daymean=np.mean([r[horizon] for r in v])
        bdays = 5 if horizon=="r5" else 1
        if side=="long": grp=[r for r in v if r["p"]>=0.6]; short=False
        else: grp=[r for r in v if r["p"]<=0.4]; short=True
        if len(grp)<3: continue
        vals=[]
        for r in grp:
            g=r[horizon]
            if short: g=-g
            if net:
                g-=rt_cost(r["px"])*100
                if short: g-=0.2*bdays
            dm = daymean if not short else -daymean
            vals.append(g-dm)
        out.append((d,np.mean(vals)))
    return out

def gauntlet(series, label):
    if series is None or len(series)<3:
        print(f"  [{label}] insufficient days ({0 if series is None else len(series)})"); return None
    vals=np.array([x for _,x in series]); n=len(vals)
    mean=vals.mean(); sd=vals.std(ddof=1) if n>1 else 0
    t=mean/(sd/np.sqrt(n)) if sd>0 else 0
    cut=int(n*0.7); early=vals[:cut].mean(); late=vals[cut:].mean()
    order=np.argsort(-vals); keep=np.ones(n,bool); keep[order[:5]]=False
    drop5=vals[keep].mean()
    rng=np.random.default_rng(0); cnt=0; B=20000
    for _ in range(B):
        s=rng.choice([-1,1],size=n)
        if (vals*s).mean()>=mean: cnt+=1
    p=cnt/B
    surv=(n>=15) and (mean>0) and (early>0) and (late>0) and (drop5>0) and (p<0.05)
    print(f"  [{label}] days={n} mean={mean:.3f}% t={t:.2f} OOS(e={early:.3f},l={late:.3f}) "
          f"drop5={drop5:.3f} perm_p={p:.4f} SURV={surv}")
    return dict(label=label,n=n,mean=mean,t=t,early=early,late=late,drop5=drop5,p=p,surv=surv)

print("\n=== M1: cross-sectional model_p_5d rank, 5d horizon ===")
for side in ["long_high","short_low","long_high_short_low","short_high","long_low"]:
    gauntlet(daily("r5",0.1,side), f"M1-{side}-d10-5d")
print("--- quintile (20%) ---")
for side in ["long_high","short_low","long_high_short_low"]:
    gauntlet(daily("r5",0.2,side), f"M1-{side}-q20-5d")
print("--- 1d horizon, decile ---")
for side in ["long_high","short_low","long_high_short_low"]:
    gauntlet(daily("r1",0.1,side), f"M1-{side}-d10-1d")

print("\n=== M2: model's own buy/sell threshold calls (p>=0.6 long, p<=0.4 short) ===")
for side in ["long","short"]:
    gauntlet(thresh("r5",side), f"M2-{side}-5d")
    gauntlet(thresh("r1",side), f"M2-{side}-1d")

print("\n=== M3: BORROWABLE subset px>=10 (short side realizable) ===")
for side in ["long_high","short_low","long_high_short_low","short_high"]:
    gauntlet(daily("r5",0.1,side,min_px=10.0), f"M3-{side}-d10-5d-px10")
for side in ["long","short"]:
    gauntlet(thresh("r5",side,min_px=10.0), f"M3-thr-{side}-5d-px10")

print("\n=== M4: conditioned on clean signals (source_weight>=7, factual) ===")
for side in ["long_high","short_low","long_high_short_low"]:
    gauntlet(daily("r5",0.1,side,min_sw=7.0,require_fac=True), f"M4-{side}-d10-5d-clean")

print("\n=== GROSS spread (no cost/borrow): hi-conf decile minus lo-conf decile, 5d & 1d ===")
def gross(horizon, top_frac=0.1, min_px=1.0):
    out=[]
    for d in dense_days:
        v=[r for r in byday[d] if r["px"]>=min_px]
        if len(v)<20: continue
        v=sorted(v,key=lambda r:r["p"]); k=max(1,int(len(v)*top_frac))
        hi=np.mean([r[horizon] for r in v[-k:]]); lo=np.mean([r[horizon] for r in v[:k]])
        out.append((d,hi-lo))
    return out
for h in ["r1","r5"]:
    s=gross(h); vals=np.array([x for _,x in s])
    sd=vals.std(ddof=1); t=vals.mean()/(sd/np.sqrt(len(vals))) if sd>0 else 0
    print(f"  GROSS hi-lo {h}: days={len(s)} mean={vals.mean():.3f}% t={t:.2f} (high-conf {'beats' if vals.mean()>0 else 'lags'} low-conf)")
    s2=gross(h,min_px=10.0); v2=np.array([x for _,x in s2])
    print(f"    px>=10: mean={v2.mean():.3f}% (n_days={len(s2)})")

# Pooled rank correlation (Spearman) of model_p vs forward return, within-day demeaned
print("\n=== Within-day Spearman(model_p, demeaned r5) pooled ===")
from scipy import stats
allp=[]; allr=[]
for d in dense_days:
    v=byday[d]
    if len(v)<20: continue
    dm=np.mean([r["r5"] for r in v])
    for r in v:
        allp.append(r["p"]); allr.append(r["r5"]-dm)
rho,pp=stats.spearmanr(allp,allr)
print(f"  pooled within-day-demeaned Spearman rho={rho:.4f} p={pp:.4g} (n={len(allp)})")
