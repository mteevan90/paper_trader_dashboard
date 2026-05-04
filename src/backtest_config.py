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

    def to_dict(self) -> dict:
        """Return a JSON-serializable dict for the experiment log."""
        return asdict(self)
