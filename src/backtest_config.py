"""backtest_config.py - Centralized backtest configuration.

Frozen dataclass holding every parameter that defines a single backtest run.
Optuna will vary the tunable subset across trials in segment 5; the fixed
subset stays constant for reproducibility. ``run_backtest()`` takes one of
these as input, so any sufficiently-detailed run description is fully
captured by a serialized ``BacktestConfig``.

TODO(v2): The current weighting scheme treats ``weight_fundamental``,
``weight_technical``, and ``weight_model`` as three free parameters and
derives ``weight_alt = 1.0 - sum_of_others`` in ``__post_init__``. An
alternative under consideration is all-four-free-with-constraint-
renormalization (Optuna picks 4 raw values, we normalize so they sum to
1.0). Three-free-plus-derived is the v1 design because it keeps the tunable
count low and the failure mode obvious. Revisit if Optuna results suggest
the alt slot is genuinely useful.
"""

from dataclasses import asdict, dataclass


# Tolerance for the weights-sum-to-1.0 validation. 1e-9 is comfortably
# tighter than typical user input precision (3 decimal places) but loose
# enough to absorb floating-point error from sums like 0.40 + 0.30 + 0.30
# that aren't exactly 1.0 in IEEE 754.
_WEIGHT_SUM_TOL = 1e-9


@dataclass(frozen=True)
class BacktestConfig:
    """Every parameter that defines a backtest run.

    Tunable fields are what Optuna varies across trials. Fixed fields stay
    constant for reproducibility - they're in the config so a serialized
    run is fully self-describing, not because Optuna will tune them.
    """

    # --- Tunable (Optuna search space in segment 5) ----------------------
    # Defaults updated in segment 12 (alt-bucket refactor):
    #   0.40/0.30/0.30 (no alt) → 0.35/0.25/0.25 (alt slot = 0.15 derived)
    # See alt_signals.py for the alt bucket aggregation slot. With an
    # empty alt registry the bucket returns 0.5 neutral and contributes
    # a constant offset to every composite (doesn't affect ranking),
    # but the f/t/m weight ratio shift does affect ranking at the margin.
    weight_fundamental: float = 0.35
    weight_technical:   float = 0.25
    weight_model:       float = 0.25
    # weight_alt is DERIVED in __post_init__ as 1.0 - sum_of_others.
    # User-supplied values are overridden. See module docstring TODO(v2)
    # about the alternative four-free-with-renormalization design.
    weight_alt:         float = 0.0
    macro_threshold_low:  float = 0.4    # below: 50% position sizing
    macro_threshold_high: float = 0.6    # above: 100% position sizing
    atr_multiplier:     float = 2.5
    analyst_weight:     float = 0.05
    rebalance_frequency_days: int = 21   # ~monthly
    position_count:     int = 15

    # --- Fixed (constant across Optuna trials) ---------------------------
    sector_cap:             int   = 3
    min_hold_days:          int   = 5
    earnings_blackout_days: int   = 3
    atr_floor_pct:          float = 0.05   # 5% minimum stop floor
    # 15% maximum stop-loss ceiling. Currently fixed; candidate for
    # promotion to a tunable parameter if Optuna trials show the cap is
    # binding (i.e. raw atr_multiplier * ATR stops are routinely hitting
    # -15% and getting clamped).
    atr_cap_pct:            float = 0.15
    transaction_cost_pct:   float = 0.0005  # 0.05% per trade
    starting_capital:       float = 100_000.0
    train_start:            str   = "2018-01-01"
    train_end:              str   = "2023-12-31"
    validate_start:         str   = "2024-01-01"
    validate_end:           str   = "2026-04-30"

    # --- Regime-dependent architecture (per regime_dependent_v1_spec) ----
    # When architecture == "legacy" (default), the existing single-value
    # tunables above are used unconditionally and all *_offensive fields
    # are ignored. When architecture == "regime-dependent", the existing
    # fields hold the DEFENSIVE values and the *_offensive fields hold
    # the OFFENSIVE values; the per-rebalance macro signal is compared
    # against regime_threshold to pick which set to apply via
    # get_active_tunables(macro_signal).
    architecture: str = "legacy"
    regime_threshold:                    float | None = None
    weight_fundamental_offensive:        float | None = None
    weight_technical_offensive:          float | None = None
    weight_model_offensive:              float | None = None
    weight_alt_offensive:                float | None = None
    atr_multiplier_offensive:            float | None = None
    position_count_offensive:            int | None = None
    rebalance_frequency_days_offensive:  int | None = None

    # --- Single-regime mode (V2 Track B control) -----------------------
    # When True, get_active_tunables ALWAYS returns the offensive
    # tunable set regardless of macro_signal — the regime architecture
    # is structurally present (offensive_* fields hold the tunables) but
    # never branches at runtime. regime_threshold is not required and
    # has no effect. Used by Track B of the V2 study to ablate the
    # regime-switching mechanism while keeping every other tunable +
    # validation rule identical to Track A. Requires architecture =
    # 'regime-dependent' so the offensive_* field set is in scope.
    single_regime_mode: bool = False

    # --- Continuous-sizing mode (opt-in, V3 study) ---------------------
    # When BOTH high_macro_threshold and low_macro_threshold are set,
    # backtest.compute_target_position_count() interpolates the rebalance
    # position count between 5 (signal >= high) and 15 (signal <= low)
    # at every rebalance date based on that day's macro_signal. The
    # endpoints are hardcoded in backtest.py for this study; future
    # studies can promote them to tunables. When either field is None
    # (default), behavior is unchanged — the rebalance loop uses
    # active["position_count"] as before. Constraint: high > low; the
    # search space + objective_fn enforce this.
    high_macro_threshold: float | None = None
    low_macro_threshold:  float | None = None

    def __post_init__(self) -> None:
        # Derive weight_alt from the three free weights and validate the
        # composite sums to 1.0. Frozen dataclasses disallow normal
        # attribute assignment - object.__setattr__ is the standard
        # escape hatch for setting derived fields in __post_init__.
        free_sum = (self.weight_fundamental
                    + self.weight_technical
                    + self.weight_model)
        if free_sum > 1.0 + _WEIGHT_SUM_TOL:
            raise ValueError(
                f"weight_fundamental + weight_technical + weight_model = "
                f"{free_sum} > 1.0; cannot derive non-negative weight_alt"
            )
        derived_alt = max(0.0, 1.0 - free_sum)
        object.__setattr__(self, "weight_alt", derived_alt)

        total = (self.weight_fundamental + self.weight_technical
                 + self.weight_model + self.weight_alt)
        if abs(total - 1.0) > _WEIGHT_SUM_TOL:
            raise ValueError(
                f"Composite weights sum to {total}, expected 1.0 "
                f"(tolerance {_WEIGHT_SUM_TOL})"
            )

        # --- Regime-dependent validation + offensive weight_alt derivation ---
        if self.single_regime_mode and self.architecture != "regime-dependent":
            raise ValueError(
                "single_regime_mode=True requires "
                "architecture='regime-dependent' (offensive_* fields hold "
                "the single-regime tunables)")
        if self.architecture == "regime-dependent":
            # regime_threshold is required only in two-regime mode; in
            # single_regime_mode it's never read so None is fine.
            if not self.single_regime_mode and self.regime_threshold is None:
                raise ValueError(
                    "architecture='regime-dependent' requires regime_threshold "
                    "(unless single_regime_mode=True)")
            required = {
                "weight_fundamental_offensive":        self.weight_fundamental_offensive,
                "weight_technical_offensive":          self.weight_technical_offensive,
                "weight_model_offensive":              self.weight_model_offensive,
                "atr_multiplier_offensive":            self.atr_multiplier_offensive,
                "position_count_offensive":            self.position_count_offensive,
                "rebalance_frequency_days_offensive":  self.rebalance_frequency_days_offensive,
            }
            missing = [k for k, v in required.items() if v is None]
            if missing:
                raise ValueError(
                    f"architecture='regime-dependent' requires offensive "
                    f"tunables; missing: {missing}")

            # Derive weight_alt_offensive from the 3 free offensive weights
            # (mirrors the legacy weight_alt derivation above).
            free_sum_off = (self.weight_fundamental_offensive
                            + self.weight_technical_offensive
                            + self.weight_model_offensive)
            if free_sum_off > 1.0 + _WEIGHT_SUM_TOL:
                raise ValueError(
                    f"Offensive weights sum to {free_sum_off} > 1.0; "
                    f"cannot derive non-negative weight_alt_offensive")
            derived_alt_off = max(0.0, 1.0 - free_sum_off)
            if self.weight_alt_offensive is None:
                object.__setattr__(self, "weight_alt_offensive", derived_alt_off)
            else:
                total_off = (self.weight_fundamental_offensive
                             + self.weight_technical_offensive
                             + self.weight_model_offensive
                             + self.weight_alt_offensive)
                if abs(total_off - 1.0) > _WEIGHT_SUM_TOL:
                    raise ValueError(
                        f"Offensive weights sum to {total_off}, expected 1.0")
        elif self.architecture != "legacy":
            raise ValueError(
                f"architecture must be 'legacy' or 'regime-dependent'; "
                f"got {self.architecture!r}")

    def get_active_tunables(self, macro_signal: float) -> dict:
        """Return the regime-active tunable values for this rebalance/day.

        In legacy mode the macro_signal arg is ignored — the seven tunables
        always come from the single-value fields. In regime-dependent mode,
        macro_signal < regime_threshold => DEFENSIVE (legacy fields hold
        the defensive set); >= => OFFENSIVE (use *_offensive fields).

        In single_regime_mode (V2 Track B), the offensive_* fields hold
        THE tunables and are returned unconditionally — macro_signal and
        regime_threshold are ignored. This makes the rebalance loop
        identical-shape to Track A but never exercises the switch.
        """
        if self.single_regime_mode:
            return {
                "weight_fundamental":       self.weight_fundamental_offensive,
                "weight_technical":         self.weight_technical_offensive,
                "weight_model":             self.weight_model_offensive,
                "weight_alt":               self.weight_alt_offensive,
                "atr_multiplier":           self.atr_multiplier_offensive,
                "position_count":           self.position_count_offensive,
                "rebalance_frequency_days": self.rebalance_frequency_days_offensive,
            }
        if (self.architecture != "regime-dependent"
                or self.regime_threshold is None):
            return {
                "weight_fundamental":       self.weight_fundamental,
                "weight_technical":         self.weight_technical,
                "weight_model":             self.weight_model,
                "weight_alt":               self.weight_alt,
                "atr_multiplier":           self.atr_multiplier,
                "position_count":           self.position_count,
                "rebalance_frequency_days": self.rebalance_frequency_days,
            }
        is_defensive = macro_signal < self.regime_threshold
        if is_defensive:
            return {
                "weight_fundamental":       self.weight_fundamental,
                "weight_technical":         self.weight_technical,
                "weight_model":             self.weight_model,
                "weight_alt":               self.weight_alt,
                "atr_multiplier":           self.atr_multiplier,
                "position_count":           self.position_count,
                "rebalance_frequency_days": self.rebalance_frequency_days,
            }
        return {
            "weight_fundamental":       self.weight_fundamental_offensive,
            "weight_technical":         self.weight_technical_offensive,
            "weight_model":             self.weight_model_offensive,
            "weight_alt":               self.weight_alt_offensive,
            "atr_multiplier":           self.atr_multiplier_offensive,
            "position_count":           self.position_count_offensive,
            "rebalance_frequency_days": self.rebalance_frequency_days_offensive,
        }

    def to_dict(self) -> dict:
        """Return a JSON-serializable dict for the experiment log."""
        return asdict(self)
