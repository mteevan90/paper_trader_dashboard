# Known display drift — out-of-scope cosmetic issues

Tracking a couple of display strings that became inaccurate after
segment 12's alt-bucket refactor. Neither breaks anything; both are
documentation/display drift in files that were marked don't-touch. They
get cleaned up when those files are next visited as part of the larger
backlog items already in the TODO list.

---

## 1. `dashboard.py` — legacy composite-scores HTML displays old weights

### Where
`src/dashboard.py`, function `_composite_scores_html` (~line 595):

```python
weights = f"Fundamental {int(W_FUNDAMENTAL*100)}% / Technical {int(W_TECHNICAL*100)}% / Model {int(W_MODEL*100)}%"
```

`W_FUNDAMENTAL`, `W_TECHNICAL`, `W_MODEL` are module-level aliases
imported from `backtest.py`, derived from `BacktestConfig()` defaults.

### What's wrong
After segment 12: those defaults are now 0.35 / 0.25 / 0.25, and the
composite has a new 0.15 alt slot plus the 0.05 analyst tiebreaker.
The display string shows "Fundamental 35% / Technical 25% / Model 25%"
— technically correct percentages, but missing the alt and analyst
context that now make up 20% of the composite signal.

### Why not fixed here
`dashboard.py` is in segment 12's don't-touch list. The new
`dashboard_app.py` (segment 13) reads the saved scores.json directly
and includes the "alt" key in its tab_positions table, so the live
dashboard is correct. The HTML batch report from `dashboard.py` is a
legacy path that's only invoked by the old autopilot pipeline.

### Fix when
This goes away when we extract shared compute helpers into
`src/dashboard_compute.py` (already on the TODO list per
dashboard_app.py module docstring) and have both dashboard files
consume from there. At that point `_composite_scores_html` should be
rewritten to read `BacktestConfig` fields directly and include the
alt + analyst rows.

---

## 2. `main.py` — hardcoded analyst tiebreaker weight

### Where
`src/main.py`, inside `compute_signals` (~line 444):

```python
s["composite"] = s["composite"] + 0.05 * a["analyst_score"]
```

### What's wrong
The literal `0.05` should be `config.analyst_weight`, matching what
`backtest.py` already does (`run_backtest` reads from config). Today
they happen to match (`BacktestConfig.analyst_weight = 0.05`), so
behavior is identical — but if a future segment tunes `analyst_weight`
or changes its default, `main.py`'s daily live path would silently
diverge from the backtest path.

### Why not fixed here
Pre-existing carryover from earlier segments. `main.py` is in the
don't-touch list for segment 12, and the fix is part of the broader
"migrate `main.py` to take a `BacktestConfig`" backlog item that's
already noted in `backtest.py`'s legacy-aliases comment.

### Fix when
Migrating `main.py` to the BacktestConfig-driven pattern. That segment
should also remove the legacy module-level alias block in
`backtest.py` (`INITIAL_CASH`, `TOP_N`, etc.) since `main.py` is the
last consumer of those imports.

---

## How this file gets used

This file is informational only — no code reads it. It exists so the
next person to touch `dashboard.py` or `main.py` doesn't have to
re-derive these issues from first principles. Delete or update once
both backlog items are cleaned up.
