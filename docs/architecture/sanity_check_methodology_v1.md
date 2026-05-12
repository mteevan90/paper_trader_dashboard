# Sanity-check methodology — v1

**Status:** APPROVED 2026-05-12. v1 canonical for future ML-study sanity-check decisions.
**Scope:** Permanent methodology finding from the Larger Universe v1 study, applicable to future ML-based equity studies. Covers when to run a parallel sanity-check model alongside a primary model, and when to skip it.
**Companion study:** Larger Universe v1 — see `docs/studies/larger_universe_v1/results.md`.
**Inherits from:** `docs/architecture/ml_study_cv_objectives_v1.md` (memo-shape convention).

## TL;DR

In Larger Universe v1, running ElasticNet as a parallel sanity check alongside XGBoost produced **one** substantive finding — the IC-vs-deployment-performance paradox that led to `docs/architecture/ml_study_cv_objectives_v1.md` — and otherwise duplicated information already captured by strategy-level success criteria. The single substantive finding was load-bearing for the project's methodology, but it emerged from **divergence between the two models**, not from agreement; whenever the two models agreed (as they did on drawdown failure and concentration failure), the sanity check added no information beyond the headline metrics.

**Recommendation for future ML studies: run sanity-check models on-demand to confirm strong claims, rather than as default parallel runs in every study.** Three specific triggers warrant an on-demand sanity run (high excess return, suspicious feature importance, concentration-finding investigation). The dashboard contract already supports multi-model artifacts, so on-demand runs integrate cleanly without code changes. The dual-reporting validation pattern from the CV-objectives memo applies here too — this recommendation is one data point and the triggers may revise as evidence accumulates.

## Empirical evidence — what ElasticNet contributed in Larger Universe v1

### What ElasticNet contributed that justified its inclusion

Three substantive contributions, only one of which generalizes:

1. **The XGB-CV-wins-vs-ENet-portfolio-wins paradox surfaced the CV-objective misalignment finding.** XGBoost's CV mean IC (+0.0282) was nearly double ElasticNet's (+0.0144), yet ElasticNet's Phase 4 portfolio outperformed XGBoost's by 17.7pp excess CAGR. This contradiction was the empirical entry point for the CV-objective memo: XGBoost had won CV but lost deployment, while ElasticNet had a lower CV score but a deployable portfolio. Without ElasticNet to provide the contrast, the failure mode would have looked like "XGBoost just doesn't work here" rather than "the CV objective doesn't predict deployment for top-N strategies." **This is the load-bearing finding.**
2. **ElasticNet's `alpha`-hits-search-floor behavior in Phase 3 revealed a search-space configuration limitation.** Phase 3's locked best ENet had `alpha ≈ 1e-5` — the bottom of the search range. The optimizer wanted less regularization than the search allowed. Without ENet's signal, we wouldn't have known whether the search space was a constraint. This is a process finding, not a methodology finding — surfaced via the side activity of running a second model.
3. **The DBD concentration case study required observing a model that exhibited the concentration pattern.** XGBoost's top contributor (MXL, 33.9% of alpha) failed the 25% criterion but only mildly. ElasticNet's 87.9%-DBD result was the extreme manifestation that made the feature/model/construction interaction visible. A weaker XGBoost-only result would still have failed the criterion but wouldn't have produced the case study's mechanism derivation.

### What ElasticNet contributed that strategy-level success criteria would have caught anyway

The Phase 5 honest-assessment table reports four success criteria. Without ElasticNet, three of the four would have been evaluated identically:

| Criterion | What XGBoost-only would have shown | What ElasticNet added |
|---|---|---|
| Excess CAGR vs SPY > 0 | +3.5pp test (✅ pass) | +21.2pp test for ENet (also passes, but with the caveat that 88% is DBD-driven — the caveat is the value) |
| Max DD ≤ 1.5× SPY | −33.5% XGB (❌ fail) | −37.5% ENet (also fails; consistent direction, additional confirmation) |
| No single ticker > 25% of alpha | MXL 33.9% (❌ fail, mild) | DBD 87.9% (also fails, extreme — see contribution #3 above) |
| 12-month rolling win rate ≥ 60% (soft) | 62.8% (✅ pass) | 86.7% (also passes, with the DBD caveat) |

For three of four criteria, the second model **confirmed the headline assessment without changing it**. The fourth (concentration) gained meaningfully from the second model only because the second model's concentration was extreme — i.e., the divergence in magnitude was the value, not the agreement on direction.

### The methodological insight

A sanity-check model adds value precisely when its results **diverge** from the primary model in informative ways. When both models converge on similar conclusions (as in three of four success-criteria checks here), the second model is duplicate work. When they diverge (as in the CV-vs-portfolio paradox and the concentration magnitude), the divergence is the finding.

Framed from cost: a parallel sanity-check pays its full compute and analytical-attention cost in every study regardless of whether divergence emerges. An on-demand sanity-check pays its cost only when the primary-model result is strong enough to warrant scrutiny. Over a project lifetime of 10+ studies the always-run cost compounds; the on-demand cost compounds only with strong-claim density.

## Mechanism — when divergence emerges and when it doesn't

Three patterns generate sanity-check divergence in ML studies of this shape (top-N stock-selection over a cross-section). Each is the basis for one of the three on-demand triggers below.

### Divergence pattern 1 — different model classes find different signals at the same metric

Tree models and linear models project the feature space differently. Trees capture interactions and non-linearity natively; linear models capture additive linear effects. When the underlying signal is interaction-heavy, trees outperform; when it's linear-additive, linear models outperform. The two models' CV scores can disagree on hyperparameter rankings even on the same data because they're effectively solving different optimization problems.

In Larger Universe v1, XGBoost found macro-driven date-level signal (Interpretation B in the CV memo) while ElasticNet found trend-driven ticker-level signal (the DBD pattern). Both are real signals in the data; they're different signals. The divergence wasn't "one model is wrong" — it was "the data contains multiple signals and the two model classes weight them differently."

### Divergence pattern 2 — different inductive biases produce different concentration profiles

ElasticNet's linear coefficients persistently weight trend features → strongly-trending stocks score high → those stocks get selected month after month → single-name concentration. XGBoost's tree splits route observations through different paths → less coefficient persistence → more rotation in selected names.

This is a structural property of the model classes, not a result that requires running both to discover. But the magnitude of divergence (XGB MXL 33.9% vs ENet DBD 87.9%) wasn't predictable ex ante and required observation.

### Divergence pattern 3 — same metric value, different held-out behavior

XGBoost CV +0.028 vs held-out full-IC −0.009. ElasticNet CV +0.014 vs held-out full-IC +0.040. Both models report positive in-fold CV IC; only one collapses out of fold. This is detectable with the primary model alone (compare in-fold to held-out IC), but the sanity-check model provides a comparison point that frames "how much should we expect held-out to differ from in-fold for this kind of model on this kind of data?" Without the comparison, the held-out collapse is harder to attribute.

### The three on-demand triggers

These are the conditions that warrant an on-demand sanity-check run. The thresholds are calibration starting points, refinable per study family.

1. **High excess return — primary model claims > X pp/yr excess CAGR.** Suggested threshold: 10pp. A claim this strong demands scrutiny before promotion. The sanity-check tests whether a different model class on the same data finds the same magnitude or something weaker, and what features it relies on.
2. **Suspicious feature importance — primary model's top features look like overfitting patterns.** Indicators: heavy macro-only loading, single-sector dominance, interaction-only signal with no main effects, or top features that should not have predictive power on theoretical grounds. The sanity check tests whether a model with weaker capacity (linear instead of tree, regularized instead of unregularized) finds the same signal.
3. **Concentration finding investigation — primary model concentrates on 1–3 names.** When a single name (or a small handful) drives most of the alpha, the sanity check tests whether concentration is structural to the data (different model classes converge on the same names) or specific to the primary model class (different model classes pick different names but still concentrate).

These triggers fire on **claim strength**, not on **claim favorability**. See the discipline guardrail below.

## Three options

### Option A — On-demand sanity check, conditional on triggers (recommended default)

Phase 3 and Phase 4 default to **primary model only**. Sanity-check models are not in the default sequence. When one of the three triggers above fires during Phase 5 validation (or earlier if a trigger is obvious mid-study), the study author runs a sanity-check via the on-demand script. Artifacts merge into the same study's `contract_v1/` directory; meta.json's `models[]` array is amended to include the sanity-check model with `role: "sanity_check"`.

**Tradeoffs.** Saves ~50% of tuning compute (one fewer Optuna run per study), ~30-40% of analytical attention (one fewer model's diagnostics to read). Cost: requires the discipline guardrail below (against confirmation bias on which strong claims trigger sanity checks). Risk: a finding like Larger Universe v1's CV-objective paradox might not emerge if the trigger conditions don't fire, because the paradox itself was the discovery.

**Mitigation for the missed-discovery risk.** The three triggers are calibrated to fire on the *kinds* of results that historically surfaced sanity-check findings — high excess return, macro-heavy feature importance, single-name concentration. Future studies should add a trigger if a sanity check surfaces a finding outside the current trigger set.

### Option B — Always run sanity check, accept the duplicate-work cost

Every study runs primary + sanity-check in Phase 3 and Phase 4. Cleaner data architecture (every study has the same artifacts shape). Avoids the discipline-guardrail risk.

**Tradeoffs.** Compute cost ~2x for tuning + backtest. Analytical-attention cost similar. Most of the duplicate work yields confirmation rather than information (three of four Larger Universe v1 success-criteria checks were redundant between the two models). Project compute is not free; analytical attention is the scarcer resource.

**When this is the right choice.** If strategy-level success criteria are weak (e.g., the study family doesn't have hard pass/fail criteria), always-run sanity checks are more defensible because there's less other infrastructure catching the same issues. Also appropriate for the very first study in a new family where calibration data for triggers doesn't exist yet.

### Option C — No sanity check ever, rely on strategy-level success criteria alone

Phase 3 and Phase 4 run primary model only. No sanity-check model in any study. Strategy-level success criteria are deemed sufficient to catch failure modes.

**Tradeoffs.** Lowest compute and analytical cost. No model-class comparison point ever exists, which means the project can never distinguish "data has no signal" from "this model class can't find the signal here." Findings like the Larger Universe v1 CV-objective paradox would not be surfaceable without a second model to compare against; they would manifest as "the strategy doesn't work" with no decomposition.

**When this is the right choice.** If the project has settled on a single model class as the standard and has independent evidence that other model classes don't perform comparably, the comparison-point value of a sanity check is reduced. Not appropriate for early-stage projects where the right model class is still under investigation.

### Recommendation

**Option A.** Options B and C are defensible under different project constraints — B for very-first-in-family studies where trigger calibration is unknown, C for mature projects with single-model-class commitments. For this project's current state (one ML study landed, methodology learnings accumulating, second study still being scoped), Option A is the right default.

## Implementation pattern

### Code-level: extract a reusable on-demand sanity-check runner

Future studies' default Phase 3 (`scripts/research/phase3_tuning.py` or equivalent) tunes only the primary model. The sanity-check model code stays in `src/equities/study/` but isn't invoked by the default Phase 3 driver.

Add a script `scripts/research/run_sanity_check.py`:

```python
# Invocation:
#   python scripts/research/run_sanity_check.py \
#       --study larger_universe_v1 \
#       --model elasticnet \
#       --trigger "high_excess_return: XGB claims +15pp CAGR"
#
# Reads the study's locked feature set + CV folds. Runs Optuna for the
# specified model. Writes results into the same study's contract_v1/
# directory. Appends the model to meta.json.models[] with role
# "sanity_check" and a `triggered_by` field recording the reason.
```

The trigger reason is recorded in meta.json so future readers know WHY the sanity check was run — distinguishing "we routinely run sanity checks" from "the +15pp claim triggered scrutiny."

### Contract-level: extend meta.json.models[] with the `triggered_by` field

In `docs/architecture/dashboard_contract_v1.md`, the `models` array entries currently allow `name`, `role`, and `params_path`. Add an optional field for sanity-check models:

```json
{
  "name": "elasticnet",
  "role": "sanity_check",
  "params_path": "../elasticnet_best_params.json",
  "triggered_by": "high_excess_return: XGB claims +15pp CAGR"
}
```

`triggered_by` is OPTIONAL, present only on models added via on-demand sanity check (not on the primary model, not on always-run sanity checks if a future study chooses Option B). This is an additive v1 change to the contract — schema_version stays at "v1".

### Dashboard surfacing (already supported, no code change)

The Phase 4.5 contract-conformant dashboard already renders multi-model studies. The Tuning tab's model selector defaults to `role: "primary"` (per the `feat/dashboard-primary-model-default` fix). On-demand sanity-check models appear in the selector when present, with the rich primary view shown by default and the sanity-check view one click away. No code changes needed for on-demand integration.

## Discipline guardrail — against confirmation-bias drift

The on-demand framing creates a confirmation-bias risk: **the temptation to run sanity checks only when results look good and skip them when results look bad**, because skipping is easier and rationalizable.

The discipline must be:

> **Trigger on strong claims regardless of direction.** +15pp excess CAGR triggers a sanity check (positive strong claim). −8pp excess CAGR also triggers a sanity check (negative strong claim — confirming a strategy is bad is also a strong claim worth scrutiny, because the explanation may be model-class-specific failure rather than data-specific signal absence).

The criterion is **claim strength**, not **claim favorability**. A study that finds "this strategy doesn't work" benefits from a sanity check just as much as one that finds "this strategy works" — in both cases, the question is whether the primary model's verdict is robust across model classes.

Operationalize the guardrail by writing the trigger evaluation into the Phase 5 checklist for every study, with the trigger thresholds applied symmetrically (|excess CAGR| > 10pp, not just excess CAGR > 10pp).

## Caveats

These split into two distinct epistemological concerns — whether the finding is reliable in the setting we observed it, and whether it generalizes beyond that setting.

### Reliability caveats — whether to trust the finding within this study

1. **Single-study finding.** Larger Universe v1 is one study. The "value of sanity check" assessment is partly determined ex post — we know in retrospect which findings the sanity check surfaced because we observed them. A future study with different divergence patterns might find different value.
2. **The CV-objective paradox may not repeat in v2 with corrected CV objective.** Larger Universe v1's load-bearing sanity-check contribution was the IC-vs-deployment-performance paradox. If v2 uses top-quintile-IC per the CV-objectives memo, the same paradox shouldn't emerge — which means the canonical example of sanity-check value in this memo may be a one-time finding tied to v1's specific methodology gap. The on-demand framework remains defensible for other divergence patterns, but the v1 evidence base shrinks if the paradox doesn't generalize.
3. **Trigger thresholds (10pp excess CAGR, etc.) are starting points, not validated.** The thresholds are calibrated to v1's magnitudes. Future studies in different return regimes (low-volatility, high-volatility) may need different thresholds. The dual-reporting validation pattern from the CV-objectives memo applies: log when triggers fire, log when they don't and a finding emerges anyway, and revise the thresholds empirically.

### Scope caveats — whether the finding applies beyond this setting

4. **This memo addresses ML-based stock selection studies.** Other study families (factor-based, rule-based, options strategies) may have different sanity-check needs. Factor studies typically run a regression-based sanity check that's structurally different. Rule-based studies have no model-class comparison axis to vary. The on-demand framework as stated applies to ML studies; other families need their own treatment.
5. **The recommendation assumes strategy-level success criteria are robust.** Larger Universe v1's four success criteria caught three of four failure modes without sanity-check input. A future study with weaker criteria may need the sanity check to play that role, making Option B more appropriate.
6. **The dashboard's multi-model support is project-specific.** Other projects may need different infrastructure for on-demand integration. The contract-level pattern in this memo references this project's dashboard contract v1 specifically.

## The pattern this memo establishes

Inherits the architectural-memo convention from `docs/architecture/ml_study_cv_objectives_v1.md` — filename convention, memo shape, versioning rules. This memo extends the convention with one observation: **memos that emerge in pairs** (the CV-objectives finding and this sanity-check finding both came from the same study) should reference each other explicitly. The cross-reference signals to future readers that the findings are entangled — the sanity-check value was load-bearing for the CV-objective discovery, and the CV-objective fix may reduce future sanity-check value at the v1-specific trigger.

## Open research questions

1. **Does on-demand pattern catch the same findings as always-run?** Empirical test: in v2, apply Option A. If v2 produces a result strong enough to trigger sanity check, run the sanity check, and compare findings to what an always-run sanity check would have shown. If v2 doesn't trigger, that's also data — supports either Option A (correctly skipping unnecessary work) or interpretation that the trigger calibration is too loose.
2. **What is the right threshold for "strong claim" — 10pp excess CAGR, 5pp, something else?** Calibrate across multiple studies as evidence accumulates.
3. **If v2's sanity check fails to surface novel findings, does the always-run pattern have value, or was v1's value v1-specific?** The honest answer determines whether to keep Option A as the default or revise to Option B.
4. **Should sanity-check methodology differ for "investigate suspicious result" vs "confirm promotion-worthy result"?** Same model in both cases or different? Larger Universe v1 used the same ElasticNet for both implicit roles; the divergence-as-value framing suggests they may be the same, but it's not proven.
5. **Are there divergence patterns beyond the three identified (different model classes, different inductive biases, different held-out behavior) that warrant additional triggers?** Future findings outside the current triggers should expand the trigger set rather than be ignored.

## Sourced from

- `docs/studies/larger_universe_v1/results.md` — full Larger Universe v1 writeup, including the DBD case study and the success-criteria assessment table.
- `docs/architecture/ml_study_cv_objectives_v1.md` — the methodology finding ElasticNet enabled. The current memo's load-bearing example.
- `docs/sessions/larger_universe_v1/session_log.md` — Phase 3 alpha-hits-floor finding, sanity-check role definition, the divergence-pattern observations across phases.
- `models/studies/larger_universe_v1/contract_v1/meta.json` — model role declarations (xgboost primary, elasticnet sanity_check).
- `models/studies/larger_universe_v1/contract_v1/per_ticker_attribution.parquet` — the per-model concentration profiles that drove the DBD case study mechanism.
