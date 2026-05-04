# Finnhub free-tier limits — segment 13 parked

Date documented: 2026-05-04
Status: Free tier insufficient for current validation window. Segment 13
parked. No `src/finnhub_signals.py` was created. `src/alt_signals.py`
was not modified. ALT_SIGNAL_REGISTRY remains empty.

## What we found

### `/stock/recommendation` (recommendation_trends)
Free-tier history depth is **only ~4 months** of monthly snapshots, not
~12 months as I estimated when planning segment 13.

Empirical verification — live API call on AAPL on 2026-05-04
(1 call out of the 60/min free quota):

```
HTTP 200, content-type=application/json; charset=utf-8
X-Ratelimit-Remaining: 59
X-Ratelimit-Limit:     60

Response: list of 4 monthly entries
  Most recent: period=2026-05-01, strongBuy=15, buy=24, hold=13,
               sell=2, strongSell=0
  Oldest:      period=2026-02-01, strongBuy=14, buy=21, hold=17,
               sell=2, strongSell=0
  Period range: 2026-02-01 -> 2026-05-01
  Keys: buy, hold, period, sell, strongBuy, strongSell, symbol
```

### Implication for the locked validation window
Validation = 2024-01-01 -> 2026-04-30 (28 months).
With 4-month history, the signal returns real data for only
~14% of the window (the final 2026-01-01 -> 2026-04-30 stretch).
The other 24 months would receive 0.5 neutral fallback. Expected
drift on overall validation drops below the noise floor (<0.3pp
return shift). Split-window analysis would also be uninformative
since both halves are dominated by neutral data.

### `/stock/upgrade-downgrade` — not verified
Endpoint takes explicit `from`/`to` date range and may have deeper
history. Skipped the verification call — even if history covers
2024+, the resulting signal is event-counting (noisier than rating-
consensus shifts) and still tests a free-tier proxy of an analyst
sentiment family we cannot fully validate at the free tier.

## Recommendation

Revisit Finnhub paid tier (USD $100+/month for `/stock/revisions`
EPS-revision endpoint) ONLY after free alt signals like OpenInsider
and Quiver have been individually tested and shown to contribute
alpha. EPS revisions specifically may not be worth the cost given
OpenInsider + Quiver overlap with the same analyst-sentiment
information family — both surface forward-looking institutional
conviction and would likely be substitute, not complement, signals
for analyst EPS revisions in a portfolio context.

If a future segment needs analyst-sentiment data:
  1. First check whether OpenInsider's cluster-buy signal has already
     captured the alpha that EPS revisions would have surfaced.
  2. If yes, skip Finnhub entirely.
  3. If no, evaluate the marginal cost of $100+/month vs the
     incremental alpha measured on a fresh held-out window after
     OpenInsider + Quiver are wired in.

## State after parking

- Phase 0 baseline (default config, post-segment-12 refactor,
  validation window) **remains the locked baseline**:
    +39.42% / 1.45 / -12.39% / 583 trades / 46.5% win
    alpha vs SPY annualized: -1.49pp
    score: -0.0346
- `src/finnhub_signals.py` does not exist (intentional)
- `src/alt_signals.py` ALT_SIGNAL_REGISTRY remains `[]` (empty)
- 1 API call was spent verifying the free-tier limitation; no cache
  was populated.
- Segment 14 will introduce OpenInsider as the first alt signal.

## API key handling
`FINNHUB_API_KEY` was read from `../.env` for the verification call.
The key remains valid and unused; no further calls will be made
until a future segment authorizes Finnhub access.
