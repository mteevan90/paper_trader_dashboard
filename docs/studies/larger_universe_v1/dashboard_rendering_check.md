# v1 dashboard rendering check — affected artifacts

**Status:** Findings document. No corrections applied to dashboard rendering; corrections require explicit decisions after review.
**Date:** 2026-05-13
**Companion to:** [`ic_scope_audit.md`](ic_scope_audit.md) (the audit that identified the scope issue affecting `ic_decomposition.parquet` and `decile_returns.parquet`).

## Question

Does the dashboard surface v1's scope-affected artifacts in user-facing tabs, and if so, are users seeing values described with labels that imply the standard full-cross-section interpretation when the underlying data is the held-subset interpretation?

## Method

Inventoried `src/dashboard_app.py` for any function or tab that loads `ic_decomposition.parquet` or `decile_returns.parquet`. For each, examined the user-facing labels, captions, and chart annotations to determine whether the displayed metric is described in a way that implies full-cross-section scope.

## Findings

### One affected dashboard function: `tab_contract_diagnostics` (`src/dashboard_app.py:4181-4225`)

This function renders both of v1's scope-affected artifacts in the same tab. It is loaded for every contract-conformant study via the routing in the dashboard's main sidebar; v1 (and any future contract-conformant study) displays its `ic_decomposition.parquet` and `decile_returns.parquet` through this function.

### Affected rendering #1: IC decomposition table (`src/dashboard_app.py:4181-4192`)

```python
def tab_contract_diagnostics(study_name: str) -> None:
    st.markdown("### IC decomposition")
    ic = load_contract_parquet(study_name, "ic_decomposition.parquet")
    if not ic.empty:
        st.caption(
            "Full-cross-section IC is the standard Spearman IC across all "
            "scored tickers per date, averaged. Top-quintile IC restricts "
            "to the top 20% of scores per date. For top-N portfolio "
            "strategies the top-quintile IC is the more deployment-aligned "
            "signal — see `docs/architecture/ml_study_cv_objectives_v1.md`."
        )
        st.dataframe(ic, use_container_width=True, hide_index=True)
```

**What users see:** a table with columns `model`, `full_ic_mean`, `full_ic_std`, `top_quintile_ic_mean`, `top_quintile_ic_std`, `n_dates_full`, `n_dates_top`, displayed via `st.dataframe`. For v1, the XGBoost row shows `top_quintile_ic_mean = +0.0481` (the held-subset number).

**What the caption says:** "Full-cross-section IC is the standard Spearman IC across all scored tickers per date, averaged. Top-quintile IC restricts to the top 20% of scores per date."

**The mismatch:** the caption tells users that the values come from "all scored tickers per date" (a full-cross-section description). For v1, that is not what the values measure — the values come from the cross-section restricted to held tickers. A user reading the caption and the +0.0481 number would reasonably conclude "XGBoost's top-quintile IC across the full eligible universe is +0.0481", which is not what the data shows.

This is a misleading rendering for v1, not because the chart code is wrong but because the data it loads was produced under a scope the caption does not acknowledge.

For v2 studies (and any future study using `phase5_analytics_v2.py` or equivalent), the data IS computed across the full eligible universe, and the caption is accurate. The rendering would be correct for v2-pattern data and misleading for v1-pattern data.

### Affected rendering #2: Decile returns chart (`src/dashboard_app.py:4194-4219`)

```python
    st.markdown("### Decile returns")
    dr = load_contract_parquet(study_name, "decile_returns.parquet")
    if not dr.empty:
        fig = go.Figure()
        for model in sorted(dr["model"].unique()):
            m = dr[dr["model"] == model].sort_values("decile")
            fig.add_trace(go.Bar(
                x=m["decile"].astype(int),
                y=m["mean_fwd_return"] * 100,
                name=model,
                error_y=dict(
                    type="data",
                    array=m["std_fwd_return"] * 100,
                    visible=True,
                    thickness=0.8,
                    width=0,
                ),
            ))
        fig.update_layout(
            barmode="group",
            title="Mean forward 21d return per score decile",
            xaxis_title="Decile (1 = lowest, 10 = highest)",
            yaxis_title="Mean fwd 21d return (%)",
            height=420,
        )
        st.plotly_chart(fig, use_container_width=True)
```

**What users see:** a bar chart of mean 21-day forward return per decile, with error bars showing per-decile std. For v1's XGBoost row, Decile 1 displays a bar at +35.7% with error bars representing ±202%. Other deciles cluster between +0.7% and +1.5% with much smaller error bars. The chart's visual impression is dominated by Decile 1's outlier-driven bar, which appears as an order-of-magnitude anomaly relative to the rest.

**What the chart implies:** the chart title "Mean forward 21d return per score decile" and axis label "Decile (1 = lowest, 10 = highest)" describe a per-decile distribution across "scores." There is no qualifying statement about which scope the deciles are computed over. A user looking at the chart would reasonably interpret the bars as "the average forward return of the lowest-scored tickers in the eligible universe is +35.7%."

**The mismatch:** that interpretation is what the chart visually communicates but not what v1's data measures. For v1, the +35.7% bar is the mean of ~5 held tickers per rebalance that happened to land in the bottom decile — a small-sample artifact dominated by single-ticker tail events. Under the standard full-cross-section definition, Decile 1 mean is +5.8% with std 25% — still the highest decile but with realistic dispersion and at an order-of-magnitude smaller scale.

The chart's visual impression of an order-of-magnitude anomaly in v1's bottom decile is not a real cross-sectional phenomenon. It is the consequence of a scope choice that the chart does not surface.

For v2 studies the data IS full-cross-section and the chart is accurate. As with the IC rendering, the chart code is correct; the issue is v1-specific data flowing through a chart that does not annotate scope.

### Other dashboard renderings: not affected

- **Walk-forward tab** (`tab_contract_walk_forward` at L4240+): loads `walk_forward.parquet` whose IC values are computed via `scores.merge(labels)` and are full-cross-section by construction. v2-baseline bit-reproduces v1's walk-forward IC values. Not scope-affected.
- **Per-ticker attribution** (`tab_contract_attribution` or equivalent): loads `per_ticker_attribution.parquet` which is computed over `holdings.iterrows()` and is not scope-dependent. Confirmed bit-identical at all scopes in the audit.
- **Concentration summary**, **rolling win rate**, **portfolio/holdings/trades displays**: not scope-affected (none use ticker-level price prices in their underlying analytics).

## Affected user-facing surfaces (summary)

Two specific displays inside the `tab_contract_diagnostics` tab for v1:
1. IC decomposition table showing `+0.0481` top-quintile IC for XGBoost with a caption claiming full-cross-section scope
2. Decile returns bar chart showing `+35.7%` Decile 1 mean with no scope annotation

Both displays are correct for v2-pattern data and misleading for v1-pattern data.

## What this rendering check does NOT do

- Does not modify dashboard code.
- Does not modify the v1 artifacts the dashboard loads.
- Does not assess how often the v1 study is viewed via the dashboard (would require usage analytics from the cloud Streamlit instance).
- Does not enumerate other studies that may have v1-pattern data flowing through the same tab. Other contract-conformant studies that used the same `phase5_analytics.py` pattern would have the same scope issue, but the only currently-promoted study with `ic_decomposition.parquet` / `decile_returns.parquet` artifacts is `larger_universe_v1`. (Confirm via `models/studies/*/contract_v1/` listing.)

## Candidate next steps (informational, no action taken)

If the dashboard rendering should match the underlying data's scope, candidate corrections include:

1. **Per-study scope annotation.** Add a `scope` field to `ic_decomposition.parquet` and `decile_returns.parquet` schemas (e.g., `"held_subset"` vs `"full_universe"`) and have the dashboard display the scope in the caption/title. v1's existing artifacts get `scope="held_subset"`; future studies using `phase5_analytics_v2.py` get `scope="full_universe"`. Cleanest long-term; requires a contract addition.
2. **Inline correction note on v1's dashboard view.** Add a study-specific banner on v1's diagnostics tab noting the scope issue and linking to the audit report. Minimal code change; less generalizable.
3. **Re-derive v1's affected artifacts at full-cross-section scope.** Produce parallel `.full_scope.parquet` artifacts and have the dashboard prefer the full-scope version when present. Mike explicitly declined this option in the audit-handling decision ("don't produce parallel v1 artifacts; the audit script can reproduce them on demand"). Listed here for completeness.
4. **No change.** Accept the misleading rendering for v1, on the basis that v1's writeup will carry correction notes via [`ic_scope_audit.md`](ic_scope_audit.md) and the architectural memo correction section. Users who reach the dashboard view should also have access to the writeup.

None of these are taken in this check. Decision deferred to Mike's review.
