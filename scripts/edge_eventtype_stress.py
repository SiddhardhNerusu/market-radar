"""Adversarial stress on the 3 event-type 'survivors' (leadership_change, buyback,
and insider_buy-as-reference). Tests robustness to NAME concentration (not just
day drops), winsorization of near-cap outliers, and tighter permutation.

If a thin daily basket is carried by 1-2 mega-winners (e.g. SNOW +54% in
leadership_change), dropping top NAMES should kill it.
"""
import sqlite3, numpy as np
from collections import defaultdict
DB="data/market_radar.db"
def rt_cost(px):
    if px<=0 or px<1: return 0.040
    if px<3: return 0.030
    if px<5: return 0.020
    if px<10: return 0.012
    if px<50: return 0.005
    return 0.002
con=sqlite3.connect(DB); con.row_factory=sqlite3.Row

def load(et):
    return con.execute("""
      SELECT ss.ticker tk, substr(rs.published_at,1,10) day,
             so.return_5d_pct r5, so.price_at_flag px
      FROM signal_scores ss JOIN raw_signals rs ON rs.id=ss.signal_id
      JOIN signal_outcomes so ON so.score_id=ss.id
      WHERE ss.event_type=? AND so.return_5d_pct IS NOT NULL AND COALESCE(so.data_corrupt,0)=0
        AND so.price_at_flag BETWEEN 1 AND 2000 AND ABS(so.return_5d_pct)<=100
        AND rs.published_at IS NOT NULL AND ABS(julianday(ss.scored_at)-julianday(rs.published_at))<=2
    """,(et,)).fetchall()

# universe demean (all event types, ticker-day dedup)
allrows=con.execute("""
  SELECT ss.ticker tk, substr(rs.published_at,1,10) day, so.return_5d_pct r5
  FROM signal_scores ss JOIN raw_signals rs ON rs.id=ss.signal_id
  JOIN signal_outcomes so ON so.score_id=ss.id
  WHERE so.return_5d_pct IS NOT NULL AND COALESCE(so.data_corrupt,0)=0
    AND so.price_at_flag BETWEEN 1 AND 2000 AND ABS(so.return_5d_pct)<=100
    AND rs.published_at IS NOT NULL AND ABS(julianday(ss.scored_at)-julianday(rs.published_at))<=2
""").fetchall()
uni=defaultdict(lambda:{"n":0,"s":0.0}); seen=set()
for r in allrows:
    k=(r["tk"],r["day"]);
    if k in seen: continue
    seen.add(k); uni[r["day"]]["n"]+=1; uni[r["day"]]["s"]+=r["r5"]
daymean={d:u["s"]/u["n"] for d,u in uni.items()}

def perm(vals,mean,B=20000):
    rng=np.random.default_rng(0); cnt=0
    for _ in range(B):
        s=rng.choice([-1,1],size=len(vals))
        if (vals*s).mean()>=mean: cnt+=1
    return cnt/B

def run(et, side, drop_names=0, winsor=None, min_px=1.0):
    rows=[dict(tk=r["tk"],day=r["day"],r5=r["r5"],px=r["px"]) for r in load(et) if r["px"]>=min_px]
    # dedup ticker-day
    ded=defaultdict(lambda:{"n":0,"r5":0.0,"px":0.0})
    for r in rows:
        k=(r["tk"],r["day"]); d=ded[k]; d["n"]+=1; d["r5"]+=r["r5"]; d["px"]+=r["px"]
    recs=[dict(tk=tk,day=day,r5=d["r5"]/d["n"],px=d["px"]/d["n"]) for (tk,day),d in ded.items()]
    # optional winsor of r5 to +/- cap
    if winsor is not None:
        for r in recs: r["r5"]=max(-winsor,min(winsor,r["r5"]))
    # find top contributing NAMES by total demeaned signed contribution, drop them
    if drop_names>0:
        contrib=defaultdict(float)
        for r in recs:
            g=r["r5"]-daymean[r["day"]]
            if side=="short": g=-(r["r5"])-(-daymean[r["day"]])
            contrib[r["tk"]]+=g
        worst=sorted(contrib,key=lambda k:-contrib[k])[:drop_names]
        recs=[r for r in recs if r["tk"] not in worst]
    byday=defaultdict(list)
    for r in recs: byday[r["day"]].append(r)
    out=[]
    for day,v in byday.items():
        vals=[]
        for r in v:
            g=r["r5"];
            if side=="short": g=-g
            g-=rt_cost(r["px"])*100
            if side=="short": g-=0.2*5
            dm=daymean[day] if side=="long" else -daymean[day]
            vals.append(g-dm)
        out.append(np.mean(vals))
    vals=np.array(out); n=len(vals); mean=vals.mean()
    sd=vals.std(ddof=1) if n>1 else 0; t=mean/(sd/np.sqrt(n)) if sd>0 else 0
    cut=int(n*0.7); e=vals[:cut].mean(); l=vals[cut:].mean()
    p=perm(vals,mean)
    print(f"  {et}-{side} dropN={drop_names} winsor={winsor} px>={min_px}: days={n} mean={mean:.3f}% t={t:.2f} OOS(e={e:.2f},l={l:.2f}) perm_p={p:.4f}")

for et,side in [("leadership_change","long"),("buyback","long"),("insider_buy","long")]:
    print(f"\n### {et} ({side}) ###")
    run(et,side)                       # baseline
    run(et,side,drop_names=1)          # drop single biggest name
    run(et,side,drop_names=3)          # drop top-3 names
    run(et,side,winsor=25)             # winsorize returns to +/-25%
    run(et,side,drop_names=3,winsor=25)
    run(et,side,min_px=10.0)           # borrowable/liquid
    run(et,side,drop_names=3,min_px=10.0)
