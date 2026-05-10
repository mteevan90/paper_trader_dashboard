"""Options backtest configuration (Phase 2 Section 5).

Frozen + slots dataclasses bundling all study levers into a single
immutable, serializable object. Section 6 (engine) consumes a
``BacktestConfig`` directly. Section 7 (Optuna runner) constructs one
per trial via :meth:`BacktestConfig.suggest`.

Composition over duplication: ``BacktestConfig`` embeds Section 4's
``ExitRules`` and the Section 5 ``FeeModel`` rather than redeclaring
their fields.

Validation ranges in ``__post_init__`` are deliberately wider than the
search ranges in :meth:`BacktestConfig.suggest`. Any value the trial can
emit must pass validation; the gap leaves headroom for v1.1+ studies
that widen the search without re-touching the dataclass.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from datetime import date

import optuna

from src.options.positions import ExitRules


__all__ = [
    "FeeModel",
    "BacktestConfig",
    "DEFAULT_UNIVERSE",
    "VALID_STRATEGY_CLASSES",
]


DEFAULT_UNIVERSE: tuple[str, ...] = (
    "SPX", "SPY", "QQQ",
    "AAPL", "JPM", "MSFT", "NVDA", "XOM",
)

VALID_STRATEGY_CLASSES: frozenset[str] = frozenset(
    {"covered_call", "cash_secured_put"}
)


@dataclass(frozen=True, slots=True)
class FeeModel:
    """Per-contract fee model.

    v1 defaults match Tradier Lite plan + 2026 regulatory pass-throughs.
    Broken out (broker / regulatory) rather than a flat composite so
    fee-sensitivity analysis can vary one without the other.
    """

    broker_fee_per_contract: float = 0.35
    regulatory_fee_per_contract: float = 0.10

    def __post_init__(self) -> None:
        if self.broker_fee_per_contract < 0:
            raise ValueError(
                "broker_fee_per_contract must be >= 0; "
                f"got {self.broker_fee_per_contract!r}"
            )
        if self.regulatory_fee_per_contract < 0:
            raise ValueError(
                "regulatory_fee_per_contract must be >= 0; "
                f"got {self.regulatory_fee_per_contract!r}"
            )

    def total_per_contract_one_way(self) -> float:
        return self.broker_fee_per_contract + self.regulatory_fee_per_contract

    def compute_fee(
        self, num_contracts: int, *, round_trip: bool = True
    ) -> float:
        """Total fee for opening (or opening+closing if ``round_trip``)
        a position of ``num_contracts`` contracts."""
        multiplier = 2 if round_trip else 1
        return num_contracts * self.total_per_contract_one_way() * multiplier


@dataclass(frozen=True, slots=True)
class BacktestConfig:
    """Immutable container for all study parameters.

    Section 6 engine consumes; Section 7 Optuna runner constructs via
    :meth:`suggest`. ``ExitRules`` and ``FeeModel`` are embedded by
    composition — when callers want to vary exit rules or fees they pass
    a fully-constructed replacement (see :meth:`evolve`).
    """

    # --- identity / scope ---
    study_label: str
    strategy_class: str
    universe: tuple[str, ...]

    # --- backtest window ---
    start_date: date
    end_date: date
    train_val_split_date: date

    # --- entry levers ---
    dte_target: int
    strike_selector_target_delta: float
    max_concurrent_positions: int
    earnings_window_avoid: bool
    max_loss_pct_of_portfolio: float

    # --- exit levers (composed) ---
    exit_rules: ExitRules

    # --- cost model ---
    fees: FeeModel

    # --- discipline ---
    promotable: bool = False
    random_seed: int | None = None

    # --- capital + fill model (Section 6) ---
    starting_capital: float = 100_000.0
    assumed_spread_pct: float = 0.05

    def __post_init__(self) -> None:
        if not self.study_label:
            raise ValueError("study_label must be a non-empty string")
        if self.strategy_class not in VALID_STRATEGY_CLASSES:
            raise ValueError(
                f"strategy_class must be one of "
                f"{sorted(VALID_STRATEGY_CLASSES)}; "
                f"got {self.strategy_class!r}"
            )
        if not self.universe:
            raise ValueError(
                "universe must contain at least one ticker"
            )
        if self.end_date <= self.start_date:
            raise ValueError(
                f"end_date ({self.end_date.isoformat()}) must be after "
                f"start_date ({self.start_date.isoformat()})"
            )
        if not (self.start_date < self.train_val_split_date < self.end_date):
            raise ValueError(
                "train_val_split_date must lie strictly between start_date "
                f"and end_date; got start={self.start_date.isoformat()}, "
                f"split={self.train_val_split_date.isoformat()}, "
                f"end={self.end_date.isoformat()}"
            )
        if not (10 <= self.dte_target <= 90):
            raise ValueError(
                f"dte_target must be in [10, 90]; got {self.dte_target!r}"
            )
        if not (0.0 < self.strike_selector_target_delta < 1.0):
            raise ValueError(
                "strike_selector_target_delta must be in (0.0, 1.0); "
                f"got {self.strike_selector_target_delta!r}"
            )
        if self.max_concurrent_positions < 1:
            raise ValueError(
                "max_concurrent_positions must be >= 1; "
                f"got {self.max_concurrent_positions!r}"
            )
        if not (0.0 < self.max_loss_pct_of_portfolio < 0.20):
            raise ValueError(
                "max_loss_pct_of_portfolio must be in (0.0, 0.20); "
                f"got {self.max_loss_pct_of_portfolio!r}"
            )
        if self.starting_capital <= 0:
            raise ValueError(
                "starting_capital must be > 0; "
                f"got {self.starting_capital!r}"
            )
        if not (0.0 <= self.assumed_spread_pct < 1.0):
            raise ValueError(
                "assumed_spread_pct must be in [0.0, 1.0); "
                f"got {self.assumed_spread_pct!r}"
            )

    @classmethod
    def suggest(
        cls,
        trial: optuna.Trial,
        *,
        study_label: str,
        strategy_class: str,
        start_date: date,
        end_date: date,
        train_val_split_date: date,
        universe: tuple[str, ...] | None = None,
        fees: FeeModel | None = None,
        promotable: bool = False,
        random_seed: int | None = None,
        starting_capital: float = 100_000.0,
        assumed_spread_pct: float = 0.05,
    ) -> "BacktestConfig":
        """Construct a ``BacktestConfig`` from an Optuna trial.

        Search ranges live here (cohesive: parameter definitions and
        their search bounds in the same place). Fixed values come from
        kwargs; tunable parameters are sampled from the trial.
        ``starting_capital`` and ``assumed_spread_pct`` are passed
        through (study-level parameters, not search variables).
        """
        return cls(
            study_label=study_label,
            strategy_class=strategy_class,
            universe=universe if universe is not None else DEFAULT_UNIVERSE,
            start_date=start_date,
            end_date=end_date,
            train_val_split_date=train_val_split_date,
            dte_target=trial.suggest_int("dte_target", 25, 50),
            strike_selector_target_delta=trial.suggest_float(
                "strike_selector_target_delta", 0.15, 0.40
            ),
            max_concurrent_positions=trial.suggest_int(
                "max_concurrent_positions", 3, 10
            ),
            earnings_window_avoid=trial.suggest_categorical(
                "earnings_window_avoid", [True, False]
            ),
            max_loss_pct_of_portfolio=trial.suggest_float(
                "max_loss_pct_of_portfolio", 0.01, 0.04
            ),
            exit_rules=ExitRules(
                profit_target_pct=trial.suggest_float(
                    "profit_target_pct", 0.25, 0.80
                ),
                time_stop_dte=trial.suggest_int("time_stop_dte", 7, 28),
                stop_loss_pct=trial.suggest_float("stop_loss_pct", 1.5, 3.5),
            ),
            fees=fees if fees is not None else FeeModel(),
            promotable=promotable,
            random_seed=random_seed,
            starting_capital=starting_capital,
            assumed_spread_pct=assumed_spread_pct,
        )

    def to_dict(self) -> dict:
        """Serialize to a JSON-safe dict for snapshot reproducibility.

        Dates → ISO ``YYYY-MM-DD`` strings. ``universe`` → list (JSON
        has no tuple). ``ExitRules`` and ``FeeModel`` flattened to
        dicts.
        """
        return {
            "study_label": self.study_label,
            "strategy_class": self.strategy_class,
            "universe": list(self.universe),
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "train_val_split_date": self.train_val_split_date.isoformat(),
            "dte_target": self.dte_target,
            "strike_selector_target_delta": self.strike_selector_target_delta,
            "max_concurrent_positions": self.max_concurrent_positions,
            "earnings_window_avoid": self.earnings_window_avoid,
            "max_loss_pct_of_portfolio": self.max_loss_pct_of_portfolio,
            "exit_rules": {
                "profit_target_pct": self.exit_rules.profit_target_pct,
                "time_stop_dte": self.exit_rules.time_stop_dte,
                "stop_loss_pct": self.exit_rules.stop_loss_pct,
            },
            "fees": {
                "broker_fee_per_contract": self.fees.broker_fee_per_contract,
                "regulatory_fee_per_contract": (
                    self.fees.regulatory_fee_per_contract
                ),
            },
            "promotable": self.promotable,
            "random_seed": self.random_seed,
            "starting_capital": self.starting_capital,
            "assumed_spread_pct": self.assumed_spread_pct,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "BacktestConfig":
        """Reverse of :meth:`to_dict`. Reconstructs nested ``ExitRules``
        and ``FeeModel``."""
        return cls(
            study_label=data["study_label"],
            strategy_class=data["strategy_class"],
            universe=tuple(data["universe"]),
            start_date=date.fromisoformat(data["start_date"]),
            end_date=date.fromisoformat(data["end_date"]),
            train_val_split_date=date.fromisoformat(
                data["train_val_split_date"]
            ),
            dte_target=data["dte_target"],
            strike_selector_target_delta=data["strike_selector_target_delta"],
            max_concurrent_positions=data["max_concurrent_positions"],
            earnings_window_avoid=data["earnings_window_avoid"],
            max_loss_pct_of_portfolio=data["max_loss_pct_of_portfolio"],
            exit_rules=ExitRules(**data["exit_rules"]),
            fees=FeeModel(**data["fees"]),
            promotable=data.get("promotable", False),
            random_seed=data.get("random_seed"),
            starting_capital=data.get("starting_capital", 100_000.0),
            assumed_spread_pct=data.get("assumed_spread_pct", 0.05),
        )

    def evolve(self, **changes) -> "BacktestConfig":
        """Return a new ``BacktestConfig`` with the given fields replaced.

        Wraps :func:`dataclasses.replace`. When changing ``exit_rules``
        or ``fees`` the caller passes a fully-constructed replacement —
        no deep-merge logic.
        """
        return dataclasses.replace(self, **changes)
