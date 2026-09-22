"""Gauntlet: per-EVENT-TYPE post-event drift (the discrete catalyst channels NOT
explicitly gauntleted as standalone cross-sections in prior work).

For each event_type with >=15 live-timed days, build a daily EQUAL-WEIGHT basket
of that event's names, day-demeaned against the full live-timed universe of that
day (strip beta), net of price-bucketed round-trip cost (+borrow if short).
LONG if raw drift positive, SHORT if negative. Run full gauntlet.

Universe day-mean for demeaning = mean of ALL live-timed clean event names that
day (the cross-section), so the demean strips the market/regime move.

Gauntlet: DEDUP, DAY-DEMEAN, NET cost, OOS 70/30, DROP-TOP-5, PERM p<0.05, >=15 days.
Bonferroni: ~240+ prior tests -> lone p<0.05 is NOISE; need ~p<0.0002.
"""
import sqlite3, numpy as np
from collections import defaultdict

DB="data/market_radar.db"
def rt_cost(px):
    if px<=0: return 0.040
    if px<1: return 0.040
    if px<3: return 0.030
    if px<5: return 0.020
    if px<10: return 0.012
    if px<50: return 0.005
    return 0.002

con=sqlite3.connect(DB); con.row_factory=sqlite3.Row
rows=con.execute("""
  SELECT ss.ticker tk, ss.event_type et, substr(rs.published_at,1,10) day,
         so.return_5d_pct r5, so.return_1d_pct r1, so.price_at_flag px
  FROM signal_scores ss JOIN raw_signals rs ON rs.id=ss.signal_id
  JOIN signal_outcomes so ON so.score_id=ss.id
  WHERE so.return_5d_pct IS NOT NULL AND COALESCE(so.data_corrupt,0)=0
    AND so.price_at_flag BETWEEN 1 AND 2000 AND ABS(so.return_5d_pct)<=100
    AND rs.published_at IS NOT NULL AND ABS(julianday(ss.scored_at)-julianday(rs.published_at))<=2
""").fetchall()

# DEDUP one bet per (ticker,event_type,day)
ded=defaultdict(lambda:{"n":0,"r5":0.0,"r1":0.0,"px":0.0})
for r in rows:
    k=(r["tk"],r["et"],r["day"]); d=ded[k]; d["n"]+=1
    d["r5"]+=r["r5"]; d["r1"]+= r["r1"] if r["r1"] is not None else 0.0; d["px"]+=r["px"]
recs=[{"tk":tk,"et":et,"day":day,"r5":d["r5"]/d["n"],"r1":d["r1"]/d["n"],"px":d["px"]/d["n"]}
      for (tk,et,day),d in ded.items()]

# universe day-mean (across ALL event types, deduped to ticker-day) for demeaning
uni=defaultdict(lambda:{"n":0,"r5":0.0,"r1":0.0})
seen=set()
for r in recs:
    kk=(r["tk"],r["day"])
    if kk in seen: continue
    seen.add(kk)
    u=uni[r["day"]]; u["n"]+=1; u["r5"]+=r["r5"]; u["r1"]+=r["r1"]
daymean={d:{"r5":u["r5"]/u["n"],"r1":u["r1"]/u["n"]} for d,u in uni.items()}

by_et_day=defaultdict(lambda:defaultdict(list))
for r in recs: by_et_day[r["et"]][r["day"]].append(r)

def basket(et, horizon, side, min_px=1.0, net=True):
    out=[]
    bdays=5 if horizon=="r5" else 1
    for day,v in by_et_day[et].items():
        v=[r for r in v if r["px"]>=min_px]
        if not v: continue
        dm=daymean[day][horizon]
        vals=[]
        for r in v:
            g=r[horizon]
            if side=="short": g=-g
            if net:
                g-=rt_cost(r["px"])*100
                if side=="short": g-=0.2*bdays
            d2 = dm if side=="long" else -dm
            vals.append(g-d2)
        out.append((day,np.mean(vals)))
    return out

def gauntlet(series,label):
    if series is None or len(series)<3:
        print(f"  [{label}] insufficient days"); return None
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
    surv=(n>=15)and(mean>0)and(early>0)and(late>0)and(drop5>0)and(p<0.05)
    print(f"  [{label}] days={n} mean={mean:.3f}% t={t:.2f} OOS(e={early:.3f},l={late:.3f}) drop5={drop5:.3f} perm_p={p:.4f} SURV={surv}")
    return dict(label=label,n=n,mean=mean,t=t,early=early,late=late,drop5=drop5,p=p,surv=surv)

# event types with >=15 live-timed days, raw direction chooses side
targets={
 "analyst_upgrade":"long","insider_buy":"long","activist_position":"long","buyback":"long",
 "leadership_change":"long","routine_prospectus":"long","analyst_downgrade":"short",
 "fda_approval":"short","ipo_registration":"short","insider_sell":"short",
 "guidance_raise":"short","earnings_miss":"short","m_a_announcement":"short",
 "earnings_beat":"short","macro":"short","lawsuit":"short","dividend":"short",
}
print("=== per-event-type drift, 5d (side = raw-mean direction), full universe ===")
for et,side in targets.items():
    gauntlet(basket(et,"r5",side), f"{et}-{side}-5d")
print("\n=== same, BORROWABLE px>=10 (shorts realizable / longs cheaper-cost) ===")
for et,side in targets.items():
    gauntlet(basket(et,"r5",side,min_px=10.0), f"{et}-{side}-5d-px10")
print("\n=== 1d horizon ===")
for et,side in targets.items():
    gauntlet(basket(et,"r1",side), f"{et}-{side}-1d")
