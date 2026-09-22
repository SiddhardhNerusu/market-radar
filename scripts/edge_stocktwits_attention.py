"""Gauntlet: stocktwits social-attention edge (UNEXPLORED).

Hypotheses:
  A) Cross-sectional mention RANK (within day) predicts forward return.
     - momentum: long high-mention, short low-mention
     - fade: opposite
  B) Mention SURGE vs trailing baseline (z of log mentions) predicts forward return.
  C) Sentiment-weighted: high mentions + bullish avg sentiment.

Tradeable framing: signal known at END of day t (mentions accumulate over day t),
trade at next-day open proxy. We DO NOT have next-open price; price_at_flag is the
intraday price at time of post. The cleanest available proxy that avoids same-day
look-ahead is to use return_5d / return_1d which run FORWARD from price_at_flag.
Same-day contamination caveat is flagged.

Gauntlet: DEDUP (one bet/ticker-day), DAY-DEMEAN, NET cost (price-bucketed; +borrow
for shorts), OOS 70/30 by day, DROP-TOP-5, PERMUTATION p<0.05, >=15 distinct days.
"""
import sqlite3, numpy as np, sys
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
         so.price_at_flag AS px, ss.sentiment AS sent
  FROM signal_scores ss
  JOIN raw_signals rs ON rs.id=ss.signal_id
  JOIN signal_outcomes so ON so.score_id=ss.id
  WHERE rs.source='stocktwits_trending'
    AND so.return_5d_pct IS NOT NULL
    AND COALESCE(so.data_corrupt,0)=0
    AND so.price_at_flag BETWEEN 1 AND 2000
    AND ABS(so.return_5d_pct)<=100
""").fetchall()

# Build (ticker, day) panel: mention count + mean fwd return + mean px + mean sent
panel = defaultdict(lambda: {"n":0,"r1":0.0,"r5":0.0,"px":0.0,"sent":0.0})
for r in rows:
    k=(r["ticker"], r["day"])
    p=panel[k]; p["n"]+=1; p["r1"]+=r["r1"] if r["r1"] is not None else 0.0
    p["r5"]+=r["r5"]; p["px"]+=r["px"]; p["sent"]+=r["sent"] if r["sent"] is not None else 0.0
recs=[]
for (tk,day),p in panel.items():
    n=p["n"]
    recs.append({"ticker":tk,"day":day,"mentions":n,
                 "r1":p["r1"]/n,"r5":p["r5"]/n,"px":p["px"]/n,"sent":p["sent"]/n})

# keep only days with real breadth (>=20 tickers) for cross-sectional ranking
byday=defaultdict(list)
for r in recs: byday[r["day"]].append(r)
dense_days=sorted(d for d,v in byday.items() if len(v)>=20)
print(f"total ticker-days={len(recs)}  dense days(>=20 tk)={len(dense_days)}: {dense_days}")

def daily_longshort(horizon, top_frac, side, use_sent=False, min_px=1.0, net=True):
    """Return list of (day, daily_mean_demeaned_net_return_pct) for the long-short
    or long-only basket. side='long_high' longs high-mention; 'short_high' shorts."""
    out=[]
    for d in dense_days:
        v=[r for r in byday[d] if r["px"]>=min_px]
        if len(v)<20: continue
        # cross-sectional rank by mentions (optionally sentiment-weighted)
        key=(lambda r: r["mentions"]*(1+max(r["sent"],0))) if use_sent else (lambda r: r["mentions"])
        v=sorted(v,key=key)
        k=max(1,int(len(v)*top_frac))
        ret=horizon
        day_mean=np.mean([r[ret] for r in v])  # day cohort mean (for demean)
        hi=v[-k:]; lo=v[:k]
        def basket_net(group, short):
            vals=[]
            for r in group:
                g=r[ret]
                if short: g=-g
                if net:
                    g-= rt_cost(r["px"])*100
                    if short: g-= 0.2*5  # 5-day borrow ~0.2%/day
                vals.append(g - (day_mean if not short else -day_mean))  # day-demean
            return np.mean(vals)
        if side=="long_high":
            out.append((d, basket_net(hi, False)))
        elif side=="short_high":
            out.append((d, basket_net(hi, True)))
        elif side=="long_low":
            out.append((d, basket_net(lo, False)))
        elif side=="short_low":
            out.append((d, basket_net(lo, True)))
    return out

def gauntlet(series, label):
    if len(series)<3:
        print(f"  [{label}] insufficient days ({len(series)})"); return
    days=[d for d,_ in series]; vals=np.array([x for _,x in series])
    n=len(vals); mean=vals.mean(); t=mean/(vals.std(ddof=1)/np.sqrt(n)) if vals.std()>0 else 0
    # OOS 70/30 by day order
    cut=int(n*0.7); early=vals[:cut].mean(); late=vals[cut:].mean()
    # drop top-5 days by abs contribution
    order=np.argsort(-vals); keep=np.ones(n,bool); keep[order[:5]]=False
    drop5=vals[keep].mean()
    # permutation: sign-flip per day
    rng=np.random.default_rng(0); cnt=0; B=20000
    for _ in range(B):
        s=rng.choice([-1,1],size=n)
        if (vals*s).mean()>=mean: cnt+=1
    p=cnt/B
    surv = (n>=15) and (mean>0) and (early>0) and (late>0) and (drop5>0) and (p<0.05)
    print(f"  [{label}] days={n} mean={mean:.3f}% t={t:.2f} OOS(early={early:.3f},late={late:.3f}) "
          f"drop5={drop5:.3f} perm_p={p:.4f} SURVIVES={surv}")
    return dict(label=label,n=n,mean=mean,t=t,early=early,late=late,drop5=drop5,p=p,surv=surv)

print("\n=== Hypothesis A: cross-sectional mention rank, 5d horizon, top/bottom decile ===")
for side in ["long_high","short_high","long_low","short_low"]:
    gauntlet(daily_longshort("r5",0.1,side), f"A-{side}-d10-5d")
print("\n--- top/bottom 20%, 5d ---")
for side in ["long_high","short_high"]:
    gauntlet(daily_longshort("r5",0.2,side), f"A-{side}-q20-5d")
print("\n--- 1d horizon, decile ---")
for side in ["long_high","short_high","long_low","short_low"]:
    gauntlet(daily_longshort("r1",0.1,side), f"A-{side}-d10-1d")
print("\n--- borrowable only (px>=10), short high mention, 5d & 1d ---")
gauntlet(daily_longshort("r5",0.1,"short_high",min_px=10.0), "A-short_high-d10-5d-px10")
gauntlet(daily_longshort("r1",0.1,"short_high",min_px=10.0), "A-short_high-d10-1d-px10")

print("\n=== Hypothesis C: sentiment-weighted mention rank, 5d ===")
for side in ["long_high","short_high"]:
    gauntlet(daily_longshort("r5",0.1,side,use_sent=True), f"C-{side}-d10-5d-sent")

print("\n=== GROSS cross-sectional tilt (no cost, no borrow) high vs low mention, 5d & 1d ===")
def gross_spread(horizon, top_frac=0.1, min_px=1.0):
    out=[]
    for d in dense_days:
        v=[r for r in byday[d] if r["px"]>=min_px]
        if len(v)<20: continue
        v=sorted(v,key=lambda r:r["mentions"])
        k=max(1,int(len(v)*top_frac))
        hi=np.mean([r[horizon] for r in v[-k:]]); lo=np.mean([r[horizon] for r in v[:k]])
        out.append((d, hi-lo))  # high minus low
    return out
for h in ["r1","r5"]:
    s=gross_spread(h); vals=np.array([x for _,x in s])
    print(f"  GROSS high-low {h}: days={len(s)} mean(high-low)={vals.mean():.3f}% (high {'beats' if vals.mean()>0 else 'lags'} low)")
    # borrowable subset
    s2=gross_spread(h,min_px=10.0); v2=np.array([x for _,x in s2])
    print(f"    px>=10 subset: mean(high-low)={v2.mean():.3f}%")

print("\n=== Hypothesis B: mention SURGE vs trailing baseline (per-ticker z of log mentions) ===")
# build per-ticker time series of mentions across dense_days; need >=3 prior obs for baseline
import math
tick_series=defaultdict(dict)
for r in recs:
    tick_series[r["ticker"]][r["day"]]=r
surge_recs=[]
for tk,days_map in tick_series.items():
    ds=[d for d in dense_days if d in days_map]
    if len(ds)<4: continue
    logm=[math.log(days_map[d]["mentions"]+1) for d in ds]
    for i in range(3,len(ds)):
        base=logm[max(0,i-5):i]
        mu=np.mean(base); sd=np.std(base)
        if sd<1e-6: continue
        z=(logm[i]-mu)/sd
        rr=days_map[ds[i]]
        surge_recs.append({"day":ds[i],"z":z,"r1":rr["r1"],"r5":rr["r5"],"px":rr["px"]})
print(f"  surge obs={len(surge_recs)}")
byday_s=defaultdict(list)
for r in surge_recs: byday_s[r["day"]].append(r)
def surge_basket(horizon, side, thr=1.0, net=True):
    out=[]
    for d in sorted(byday_s):
        v=byday_s[d]
        if len(v)<10: continue
        daymean=np.mean([r[horizon] for r in v])
        hot=[r for r in v if r["z"]>=thr]
        if not hot: continue
        vals=[]
        for r in hot:
            g=r[horizon]
            if side=="short": g=-g
            if net:
                g-=rt_cost(r["px"])*100
                if side=="short": g-=(0.2*(5 if horizon=="r5" else 1))
            dm = daymean if side=="long" else -daymean
            vals.append(g-dm)
        out.append((d,np.mean(vals)))
    return out
for h in ["r1","r5"]:
    for side in ["long","short"]:
        gauntlet(surge_basket(h,side), f"B-surge>=1z-{side}-{h}")
