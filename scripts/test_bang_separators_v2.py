#!/usr/bin/env python3
"""
Bang-separator study v2 -- PRE-REGISTERED grid test on event_features.

Tests whether the literature's ex-ante separators (float proxy, prior-day RVOL,
opening gap) separate +50% 5d "bangs" from <=-20% "crashes" in the live-timed
MARKET RADAR event window (2026-05-10..2026-06-10).

Grid (pre-registered, NO cells beyond these):
  1. float<20M alone
  2. prior_rvol>=3 alone
  3. gap>=+5% alone
  4. float<20M AND prior_rvol>=3
  5. float<20M AND gap>=+5%
  6. float<20M AND prior_rvol>=3 AND gap>=+5%
Plus the same grid with eod_rvol swapped for prior_rvol (cells 2,4,6 only --
cells 1,3,5 contain no rvol term so they are byte-identical). eod_rvol cells
are FLAGGED: end-of-day knowledge, partially look-ahead for an intraday entry.

Weighting: bangs and crashes are the FULL population; meh controls are a
deterministic 1-in-40 stride sample of 9,794 population meh events, so each
sampled meh row carries weight 9794/245 = 39.976 in every rate/expectancy.

Costs: round-trip 3.5% (px<5), 1.5% (5<=px<=20), 0.6% (px>20), px = price_at_flag.

Read-only on data/research_bars.db. Deterministic (no RNG anywhere).
"""

import os
import sqlite3
from collections import namedtuple

HERE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(HERE, "..", "data", "research_bars.db")

MEH_POP = 9794          # population meh count behind the stride sample
MEH_SAMPLE = 245        # sampled meh rows in event_features
MEH_W = MEH_POP / MEH_SAMPLE

Ev = namedtuple("Ev", "ticker event_day label r5 px so so_stale prior_rvol gap_pct eod_rvol")


def cost_pct(px):
    if px is None:
        return 1.5  # should not happen (universe requires px 1..50); mid bucket
    if px < 5:
        return 3.5
    if px <= 20:
        return 1.5
    return 0.6


def w(ev):
    return MEH_W if ev.label == "meh" else 1.0


def load():
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    cur = con.execute(
        "SELECT ticker, event_day, label, r5, px, so, so_stale_flag, "
        "prior_rvol, gap_pct, eod_rvol FROM event_features "
        "ORDER BY ticker, event_day"
    )
    evs = [Ev(*row) for row in cur.fetchall()]
    con.close()
    return evs


def wstats(evs):
    """Weighted bang/crash rate (%), gross mean r5, net expectancy after cost."""
    tw = sum(w(e) for e in evs)
    if tw == 0:
        return None
    bang_w = sum(w(e) for e in evs if e.label == "bang")
    crash_w = sum(w(e) for e in evs if e.label == "crash")
    gross = sum(w(e) * e.r5 for e in evs) / tw
    net = sum(w(e) * (e.r5 - cost_pct(e.px)) for e in evs) / tw
    return dict(
        tw=tw,
        bang_rate=100.0 * bang_w / tw,
        crash_rate=100.0 * crash_w / tw,
        gross=gross,
        net=net,
    )


def label_counts(evs):
    d = {"bang": 0, "crash": 0, "meh": 0}
    for e in evs:
        d[e.label] += 1
    return d


RULES = [
    # (name, required non-null fields, predicate, ex_ante_clean)
    ("float<20M",
     ("so",), lambda e: e.so < 20e6, True),
    ("prior_rvol>=3",
     ("prior_rvol",), lambda e: e.prior_rvol >= 3.0, True),
    ("gap>=+5%",
     ("gap_pct",), lambda e: e.gap_pct >= 5.0, True),
    ("float<20M & prior_rvol>=3",
     ("so", "prior_rvol"), lambda e: e.so < 20e6 and e.prior_rvol >= 3.0, True),
    ("float<20M & gap>=+5%",
     ("so", "gap_pct"), lambda e: e.so < 20e6 and e.gap_pct >= 5.0, True),
    ("float<20M & prior_rvol>=3 & gap>=+5%",
     ("so", "prior_rvol", "gap_pct"),
     lambda e: e.so < 20e6 and e.prior_rvol >= 3.0 and e.gap_pct >= 5.0, True),
    # ---- eod_rvol swap (FLAGGED: end-of-day knowledge, partial look-ahead) ----
    ("[EOD-FLAGGED] eod_rvol>=3",
     ("eod_rvol",), lambda e: e.eod_rvol >= 3.0, False),
    ("[EOD-FLAGGED] float<20M & eod_rvol>=3",
     ("so", "eod_rvol"), lambda e: e.so < 20e6 and e.eod_rvol >= 3.0, False),
    ("[EOD-FLAGGED] float<20M & eod_rvol>=3 & gap>=+5%",
     ("so", "eod_rvol", "gap_pct"),
     lambda e: e.so < 20e6 and e.eod_rvol >= 3.0 and e.gap_pct >= 5.0, False),
]


def main():
    evs = load()
    all_days = sorted({e.event_day for e in evs})
    median_day = all_days[len(all_days) // 2]  # global split point (early < median_day cutoff)
    # early = first half of distinct days, late = rest
    early_days = set(all_days[: len(all_days) // 2 + len(all_days) % 2])

    base = wstats(evs)
    print(f"Loaded {len(evs)} events, labels={label_counts(evs)}, "
          f"{len(all_days)} distinct event-days ({all_days[0]}..{all_days[-1]})")
    print(f"meh weight = {MEH_W:.3f}")
    print(f"BASE (all events, weighted): bang_rate={base['bang_rate']:.3f}%  "
          f"crash_rate={base['crash_rate']:.3f}%  gross r5={base['gross']:+.2f}%  "
          f"net={base['net']:+.2f}%")
    print()

    results = []
    for name, req, pred, clean in RULES:
        eligible = [e for e in evs if all(getattr(e, f) is not None for f in req)]
        elig_base = wstats(eligible)
        passing = [e for e in eligible if pred(e)]
        lc_elig = label_counts(eligible)
        lc = label_counts(passing)
        st = wstats(passing)

        print(f"=== {name}  ({'EX-ANTE CLEAN' if clean else 'EOD KNOWLEDGE - FLAGGED'}) ===")
        print(f"  eligible n={len(eligible)} {lc_elig}  "
              f"eligible base bang_rate={elig_base['bang_rate']:.3f}%")
        if st is None:
            print("  n=0 pass -- empty cell")
            results.append(dict(rule=name, n=0, clean=clean))
            print()
            continue

        lift = st["bang_rate"] / elig_base["bang_rate"] if elig_base["bang_rate"] > 0 else float("nan")
        days = sorted({e.event_day for e in passing})
        print(f"  PASS n={len(passing)} {lc}  distinct_days={len(days)}")
        print(f"  bang_rate={st['bang_rate']:.3f}%  lift={lift:.2f}x  "
              f"crash_rate={st['crash_rate']:.3f}%")
        print(f"  gross mean r5={st['gross']:+.2f}%  NET expectancy={st['net']:+.2f}%")

        # early/late split (global median-day cutoff) if cell spans >=15 distinct days
        note_parts = [f"elig base {elig_base['bang_rate']:.2f}%",
                      f"raw pass b/c/m={lc['bang']}/{lc['crash']}/{lc['meh']}",
                      f"gross r5 {st['gross']:+.1f}%"]
        if len(days) >= 15:
            early = [e for e in passing if e.event_day in early_days]
            late = [e for e in passing if e.event_day not in early_days]
            se, sl = wstats(early), wstats(late)
            ed = len({e.event_day for e in early})
            ld = len({e.event_day for e in late})
            split = (f"early({ed}d,n={len(early)}): bang {se['bang_rate']:.2f}% net {se['net']:+.1f}% | "
                     f"late({ld}d,n={len(late)}): bang {sl['bang_rate']:.2f}% net {sl['net']:+.1f}%")
            print(f"  early/late split @ {median_day}: {split}")
            note_parts.append(split)
        else:
            pw = f"only {len(days)} distinct days -- too few for early/late split, low power"
            print(f"  {pw}")
            note_parts.append(pw)

        # staleness robustness for any float cell (data-quality check, not a new cell)
        if "so" in req:
            fresh_elig = [e for e in eligible if not e.so_stale]
            fresh_pass = [e for e in passing if not e.so_stale]
            fe, fp = wstats(fresh_elig), wstats(fresh_pass)
            if fp and fe and fe["bang_rate"] > 0:
                fl = fp["bang_rate"] / fe["bang_rate"]
                srob = (f"stale-excl: n={len(fresh_pass)} bang {fp['bang_rate']:.2f}% "
                        f"lift {fl:.2f}x net {fp['net']:+.1f}%")
            else:
                srob = f"stale-excl: n={len(fresh_pass)} -- too small"
            print(f"  {srob}")
            note_parts.append(srob)

        print()
        results.append(dict(
            rule=name, n=len(passing), bang_rate=st["bang_rate"], lift=lift,
            crash_rate=st["crash_rate"], gross=st["gross"], net=st["net"],
            clean=clean, days=len(days), note="; ".join(note_parts),
        ))

    return base, results


if __name__ == "__main__":
    main()
