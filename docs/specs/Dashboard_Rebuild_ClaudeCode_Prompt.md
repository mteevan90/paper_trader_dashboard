# Dashboard Rebuild — Claude Code Implementation Prompt

You're rebuilding the Streamlit dashboard at `src/dashboard_app.py` to be reader-friendly for three finance-literate non-quants (Mike, his brother, his father). The spec is attached as `Dashboard_Rebuild_Spec.docx`. Read that document first; this prompt adds one important section that supersedes/extends section 3 of the spec.

## The added requirement: exec summary on every analytical tab

Every tab in the new structure (Performance, Current Holdings, Trade History, Market Context, Risk & Behavior, Reliability, Tuning History) gets an exec summary at the very top of the page. Glossary & Help is the only tab without one — it's a reference page, no summary needed.

The exec summary sits ABOVE Layer 1 (the quick-inference cards). The new top-to-bottom order on each tab becomes:

```
PAGE TITLE
├── Exec summary  ← NEW (st.info or styled blockquote)
├── divider
├── Layer 1 — Quick inference (large metric cards / hero chart)
├── divider
├── Layer 2 — Visual breakdown
├── divider
└── Layer 3 — Detailed view (some in expanders)
```

## Why this matters

Without exec summaries, readers face cold KPI cards with no narrative. Even a finance person reading "Total return +315.7%" needs context — is that good? was the period favorable? what should I worry about? The summary provides that narrative in 3-5 sentences, then the KPIs confirm the same story numerically.

Crucially, **the summary's signals must agree with the KPIs and charts below it.** If the summary says "strong outperformance" but the KPI shows +5pp alpha, the reader gets confused and loses trust. To prevent this, summaries are **data-driven** (generated from the same numbers shown in the KPIs), not hardcoded prose.

## Summary structure (mandatory)

Every exec summary has exactly this structure, in this order:

1. **Headline (1-2 sentences):** What this tab is showing and the top-line answer.
2. **Detail (2-3 sentences):** The shape of the answer — what's behind the headline number, what's notable, who-dunit if there's a clear driver.
3. **Caveat (1 sentence, MANDATORY):** What's limiting, ambiguous, or worth questioning about the result. This is the "honest finance professional" part. Don't skip this. Don't soften it. Don't make it generic boilerplate that's the same on every tab.

Render the summary as `st.info()` (gives a clear visual block) or `st.markdown()` with a blockquote `> ` prefix on each line. Pick one and use it consistently. Keep the entire summary to 4-6 sentences total.

## Config awareness

Each summary is generated based on the currently-loaded result. The sidebar's "Default config" vs "Best trial of selected study" toggle changes which `meta.json` and `result` dict are loaded. The summary regenerates accordingly.

Concretely: write a function `_exec_summary_<tab_name>(label: str, config: dict, result: dict) -> str` for each tab. Call it from inside the tab function before rendering Layer 1. The function reads from `result` and `meta` and returns the summary string. This keeps summary logic isolated and testable.

## Signal-KPI consistency rules

For each tab, the summary's adjectives must come from the same numbers the KPIs display. Use threshold-based templating:

**Performance tab example:**
```python
def _exec_summary_performance(label, config, result):
    meta = result["meta"]
    components = meta.get("components", {})  # or wherever alpha lives
    alpha_arith = components.get("alpha_annualized", 0) * 100  # in pp
    total_return = components.get("strategy_total_return", 0) * 100
    spy_return = components.get("spy_total_return", 0) * 100
    max_dd = abs(components.get("max_drawdown", 0)) * 100
    
    # Strength wording driven by data
    if alpha_arith >= 30: strength = "strongly outperforms"
    elif alpha_arith >= 10: strength = "outperforms"
    elif alpha_arith >= 0: strength = "modestly beats"
    else: strength = "underperforms"
    
    # Drawdown framing
    if max_dd >= 25: dd_note = f"a meaningful -{max_dd:.0f}% peak-to-trough drop"
    elif max_dd >= 15: dd_note = f"a -{max_dd:.0f}% maximum drawdown"
    else: dd_note = f"a manageable -{max_dd:.0f}% drawdown"
    
    headline = (f"This strategy {strength} the S&P 500 over the validation "
                f"window: +{total_return:.1f}% total return vs SPY's "
                f"+{spy_return:.1f}% (annualized {alpha_arith:+.1f}pp).")
    
    detail = (f"Performance came with {dd_note} along the way, and the "
              f"strategy moves more than the market (beta around 1.3) — so "
              f"some of the outperformance reflects amplified market exposure "
              f"rather than pure stock selection.")
    
    caveat = (f"The 2024-2026 validation period was a strong bull market for "
              f"tech and quality stocks, which align with this strategy's "
              f"selection criteria. Performance in a different market regime "
              f"could be materially different.")
    
    return f"{headline}\n\n{detail}\n\n*{caveat}*"
```

That's the pattern. Adjectives ("strongly", "meaningful", "manageable") come from numerical thresholds; numbers come from the loaded data; the caveat is consistent on this tab but the wording can incorporate the loaded data where relevant (e.g., note the actual drawdown number instead of saying "drawdowns happened").

## Per-tab summary content guide

For each analytical tab, here's what the summary should say. Templates apply when "Best trial of selected study" is loaded (the typical case); for "Default config" mode, use simpler wording and emphasize that the config is unoptimized.

### Performance
- Headline: outperformance vs SPY (use the wording rules above)
- Detail: the source of outperformance (high beta + alpha, vs pure alpha) and the journey (drawdowns, recovery)
- Caveat: bull-market validation period; backtest doesn't include real-money frictions (slippage, taxes)

### Current Holdings
- Headline: how concentrated and how many sectors
- Detail: largest position by % weight; whether the portfolio is sector-concentrated
- Caveat: a 5-position portfolio carries materially more single-name risk than a diversified ETF; if any one of these names blows up, performance suffers disproportionately

### Trade History (config-aware AND date-filter-aware)
- Headline: number of trades and biggest winner/loser by ticker over the selected period
- Detail: win rate; whether activity is steady or clustered in time
- Caveat: realized P&L only — open positions at end of period are not counted; FIFO matching can attribute gains/losses imperfectly when positions are partial-sized

### Market Context
- Headline: current market read (Bullish / Mixed / Stressed) based on macro signal level
- Detail: which signal components are flagging stress vs calm; what sizing the strategy is using as a result
- Caveat: the macro signal floor in the validation period was 0.42 — meaning the "Stressed" tier never fired. The macro overlay is structurally present but has not been exercised in current data.

### Risk & Behavior
- Headline: how the strategy handles up vs down markets (capture ratios)
- Detail: drawdown depth and recovery time; consistency of monthly outperformance
- Caveat: drawdowns of -22% are real and would be psychologically difficult for a real investor; concentration amplifies volatility

### Reliability (Track 2 robustness data)
- Headline: how robust the chosen config is across small parameter perturbations
- Detail: which settings are stable, which are sensitive, what the practical implication is
- Caveat: Trial #325 sits at a TPE-found peak; some axes show ~30pp alpha drop with small parameter shifts. The strategy concept is robust; the specific peak result is sensitive.

### Tuning History
- Headline: how many configs tested and what the winner was
- Detail: progression of scores over trials; how clearly the winner emerged
- Caveat: Optuna is search, not proof. A different random seed or longer search might find a better config or might find that this peak doesn't generalize to other validation windows.

## Loading state and edge cases

- If a tab's data isn't loaded yet (cloud mode, R2 fetch in progress): the summary function returns a placeholder like "Loading summary…" or "*Summary will appear when results are loaded.*"
- If `meta.get("promoted") is False` (an experimental config): prepend the headline with "**Experimental config** — " so readers know this isn't the locked baseline.
- If the user toggles "Default config" mode: the caveat changes to "*This is the unoptimized baseline configuration, not the locked V1 strategy. Numbers shown are illustrative.*"

## Implementation order

Do the work in this order so smoke tests catch problems early:

1. Read `Dashboard_Rebuild_Spec.docx` in full.
2. Read `docs/dashboard_audit_20260507.md` to confirm the structural assumptions about main(), tab dispatch, and existing patterns.
3. Read the current `dashboard_app.py` to understand existing tab functions, loaders, and the cache layer.
4. Implement tab renames + reorder per spec section 2 (this is the smallest cost-to-test step — verify all 8 tabs still load).
5. Implement the three-layer pattern per spec section 4, tab by tab. Run AppTest after each tab to catch issues early.
6. Implement exec summaries per THIS prompt's rules. Test that toggling between Default config and Best trial regenerates the summary.
7. Apply the term translation table per spec section 5.
8. Build the Glossary tab content per spec section 4.
9. Run the full verification checklist in spec section 7.
10. Commit with the message in spec section 7.

## Don'ts

- Don't refactor data_source.py or the caching layer. The audit said the bones are healthy.
- Don't fake data the backtest doesn't emit. Per-holding sub-scores, per-ticker alpha attribution — these are deferred.
- Don't make exec summaries hardcoded strings. They must regenerate from loaded data.
- Don't skip the caveat sentence on any tab. The summary's value is its honesty.
- Don't write summaries that read like marketing copy ("our strategy delivers exceptional returns…"). Plain, professional, slightly skeptical tone.
- Don't break existing functionality from commits 3c0e671 (Robustness tab) or 37c5b49 (alpha clarification + 404 silencing). The 4-card alpha row from 37c5b49 lives in Performance Layer 3 now; preserve all four cards.

## Cost estimate

This is a meaningful Claude Code session: ~3-4 hours given the 8-tab scope plus exec summaries. Net dashboard_app.py change probably +500-700 lines (Layer 1 cards are new; exec summary functions are new; some Layer 2/3 content moves rather than grows; some duplicated boilerplate gets consolidated).

Single commit at the end, please. The commit message in spec section 7 covers the rename/reorder/three-layer changes; extend it with one bullet about the exec summaries:

> - Added per-tab exec summaries (3-5 sentences each, data-driven, with mandatory caveat sentence) at the top of every analytical tab. Glossary tab unchanged.

## Verification additions

Beyond spec section 7's checklist, verify:

- Exec summary appears at the top of all 7 analytical tabs (not on Glossary).
- Each summary has 3 distinct sections: headline, detail, caveat.
- Toggling sidebar between Default config and Best trial changes the summary text.
- For Trial #325, the Performance summary mentions outperformance (matching the +63.7pp KPI), the Reliability summary mentions some sensitivity (matching the 47% / 84% data), the Market Context summary mentions a Bullish or Mixed read (matching the current macro signal level).
- No summary contains the placeholder text "Loading summary…" when data is loaded.
