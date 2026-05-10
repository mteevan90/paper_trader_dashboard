"""Options Optuna runner (Phase 2 Section 7).

Wraps the Section 6 backtest engine in an Optuna TPE objective:

- :func:`calmar_objective` — pure function; Calmar ratio on training-
  window snapshots only. Validation data is tagged in
  :class:`StudyResults` but excluded from optimization.
- :class:`OptunaStudyResults` — frozen+slots summary of a completed
  study, with ``to_json`` / ``from_json`` for snapshotting.
- :func:`run_optuna_study` — constructs the TPE study, runs N trials,
  persists top-K trial outputs to disk, returns the summary.

SQLite storage at ``models/cache/options/optuna_studies.db`` (separate
from crypto's per-asset-class isolation). Resume via
``load_if_exists=True``. Failed trials log and return ``-1.0`` so the
study continues; the count surfaces in ``n_trials_failed``.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Optional

import optuna

from src.options.backtest_config import BacktestConfig, DEFAULT_UNIVERSE
from src.options.engine import EngineDeps, StudyResults, run_backtest


__all__ = [
    "calmar_objective",
    "OptunaStudyResults",
    "run_optuna_study",
    "DEFAULT_OPTUNA_STORAGE_PATH",
    "FAILED_TRIAL_SENTINEL",
    "ZERO_DD_POSITIVE_RETURN_SENTINEL",
]


logger = logging.getLogger(__name__)


DEFAULT_OPTUNA_STORAGE_PATH: Path = (
    Path("models") / "cache" / "options" / "optuna_studies.db"
)
FAILED_TRIAL_SENTINEL: float = -1.0
ZERO_DD_POSITIVE_RETURN_SENTINEL: float = 1e9
_MIN_TRAINING_DAYS: int = 30


def calmar_objective(results: StudyResults) -> float:
    """Calmar ratio for the training portion of a backtest.

    Calmar = annualized compound return / max drawdown. Higher is
    better; Optuna optimizes for maximization. Computed from
    ``results.daily_snapshots`` filtered to ``train_val_label == "train"``.
    Validation data is excluded — reporting-only, never used as the
    optimization target.

    Edge cases (return values picked so Optuna ranks them sensibly):

    - Empty training data → ``0.0``
    - Training window < 30 days → ``0.0`` (not enough data for Calmar)
    - Initial portfolio value ≤ 0 → ``0.0`` (defensive)
    - Complete wipeout (final ≤ 0) → ``-1.0`` from the compound-return path
    - Zero drawdown with positive return → ``1e9`` sentinel (top rank)
    - Zero drawdown with non-positive return → return the compound return

    Failed-trial pathway is the *caller's* concern: an exception during
    ``run_backtest`` should be caught upstream and reported as
    ``FAILED_TRIAL_SENTINEL`` directly, not via this function.
    """
    train_snaps = [
        s for s in results.daily_snapshots
        if s.train_val_label == "train"
    ]
    if len(train_snaps) < _MIN_TRAINING_DAYS:
        return 0.0

    initial = train_snaps[0].portfolio_total
    final = train_snaps[-1].portfolio_total
    if initial <= 0:
        return 0.0

    days = (train_snaps[-1].sim_date - train_snaps[0].sim_date).days
    years = days / 365.25
    if years <= 0:
        return 0.0

    if final <= 0:
        compound_return = -1.0
    else:
        compound_return = (final / initial) ** (1.0 / years) - 1.0

    peak = initial
    max_dd = 0.0
    for s in train_snaps:
        if s.portfolio_total > peak:
            peak = s.portfolio_total
        if peak > 0:
            dd = (peak - s.portfolio_total) / peak
            if dd > max_dd:
                max_dd = dd

    if max_dd == 0.0:
        if compound_return > 0:
            return ZERO_DD_POSITIVE_RETURN_SENTINEL
        return compound_return

    return compound_return / max_dd


@dataclass(frozen=True, slots=True)
class OptunaStudyResults:
    """Summary of a completed Optuna study."""

    study_label: str
    strategy_class: str
    n_trials_run: int
    n_trials_failed: int
    best_value: float
    best_trial_number: int
    best_params: dict[str, object]
    top_k_trial_numbers: tuple[int, ...]
    wall_time_seconds: float
    storage_path: Path
    output_dir: Path

    def to_json(self, path: Path) -> None:
        """Serialize to JSON. ``Path`` objects converted to strings;
        ``best_params`` values pass through (Optuna parameter values are
        JSON-safe primitives)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "study_label": self.study_label,
            "strategy_class": self.strategy_class,
            "n_trials_run": self.n_trials_run,
            "n_trials_failed": self.n_trials_failed,
            "best_value": self.best_value,
            "best_trial_number": self.best_trial_number,
            "best_params": dict(self.best_params),
            "top_k_trial_numbers": list(self.top_k_trial_numbers),
            "wall_time_seconds": self.wall_time_seconds,
            "storage_path": str(self.storage_path),
            "output_dir": str(self.output_dir),
        }
        with open(path, "w") as fh:
            json.dump(data, fh, indent=2, default=str)

    @classmethod
    def from_json(cls, path: Path) -> "OptunaStudyResults":
        with open(path) as fh:
            data = json.load(fh)
        return cls(
            study_label=str(data["study_label"]),
            strategy_class=str(data["strategy_class"]),
            n_trials_run=int(data["n_trials_run"]),
            n_trials_failed=int(data["n_trials_failed"]),
            best_value=float(data["best_value"]),
            best_trial_number=int(data["best_trial_number"]),
            best_params=dict(data["best_params"]),
            top_k_trial_numbers=tuple(
                int(n) for n in data["top_k_trial_numbers"]
            ),
            wall_time_seconds=float(data["wall_time_seconds"]),
            storage_path=Path(data["storage_path"]),
            output_dir=Path(data["output_dir"]),
        )


def run_optuna_study(
    *,
    study_label: str,
    strategy_class: str,
    universe: Optional[tuple[str, ...]],
    start_date: date,
    end_date: date,
    train_val_split_date: date,
    n_trials: int,
    starting_capital: float = 100_000.0,
    promotable: bool = False,
    output_dir: Optional[Path] = None,
    top_k: int = 5,
    deps: Optional[EngineDeps] = None,
    seed: int = 42,
    storage_path: Optional[Path] = None,
) -> OptunaStudyResults:
    """Run an Optuna TPE study around the Section 6 backtest engine.

    Each trial: :meth:`BacktestConfig.suggest` constructs a config from
    the trial; :func:`run_backtest` evaluates it; :func:`calmar_objective`
    scores the result. Top-K trial results (by objective value) persist
    full :class:`StudyResults` parquet to
    ``<output_dir>/trial_<NNNN>/``; the rest retain only Optuna's
    parameter-and-value summary in the SQLite study.

    Resume: ``study_label`` + ``storage_path`` identify a study uniquely.
    If both already exist on disk, Optuna picks up where it left off
    (``load_if_exists=True``).

    Failure handling: trials that raise log the exception and return
    :data:`FAILED_TRIAL_SENTINEL`. The study continues; the failure
    count surfaces in ``n_trials_failed``.
    """
    storage_path = storage_path or DEFAULT_OPTUNA_STORAGE_PATH
    storage_path = Path(storage_path)
    storage_path.parent.mkdir(parents=True, exist_ok=True)
    storage_url = f"sqlite:///{storage_path}"

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(
        study_name=study_label,
        storage=storage_url,
        sampler=sampler,
        direction="maximize",
        load_if_exists=True,
    )

    trial_results: dict[int, StudyResults] = {}
    failed_trials = 0
    t0 = time.time()

    def objective(trial: optuna.Trial) -> float:
        nonlocal failed_trials
        try:
            config = BacktestConfig.suggest(
                trial,
                study_label=study_label,
                strategy_class=strategy_class,
                universe=universe if universe is not None else DEFAULT_UNIVERSE,
                start_date=start_date,
                end_date=end_date,
                train_val_split_date=train_val_split_date,
                promotable=promotable,
                random_seed=seed,
                starting_capital=starting_capital,
            )
            results = run_backtest(config, deps=deps)
            score = calmar_objective(results)
            trial_results[trial.number] = results
            return score
        except Exception as exc:
            failed_trials += 1
            logger.error(
                "Trial %d failed: %r", trial.number, exc
            )
            return FAILED_TRIAL_SENTINEL

    study.optimize(objective, n_trials=n_trials, n_jobs=1)
    wall_time = time.time() - t0

    # Identify top-K successful trials by objective value, excluding
    # the failed-trial sentinel.
    successful = [
        t for t in study.trials
        if t.value is not None and t.value > FAILED_TRIAL_SENTINEL
        and t.number in trial_results
    ]
    successful.sort(key=lambda t: t.value, reverse=True)
    top_k_trials = successful[:top_k]
    top_k_trial_numbers = tuple(t.number for t in top_k_trials)

    # Persist top-K full StudyResults to disk.
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for trial_number in top_k_trial_numbers:
            trial_dir = output_dir / f"trial_{trial_number:04d}"
            trial_dir.mkdir(parents=True, exist_ok=True)
            trial_results[trial_number].to_parquet(trial_dir)

    # Optuna raises if best_* is accessed on a study with no successful
    # trials. Defensively report sentinel/empty if everything failed.
    if successful:
        best_value = float(study.best_value)
        best_trial_number = int(study.best_trial.number)
        best_params = dict(study.best_params)
    else:
        best_value = FAILED_TRIAL_SENTINEL
        best_trial_number = -1
        best_params = {}

    return OptunaStudyResults(
        study_label=study_label,
        strategy_class=strategy_class,
        n_trials_run=len(study.trials),
        n_trials_failed=failed_trials,
        best_value=best_value,
        best_trial_number=best_trial_number,
        best_params=best_params,
        top_k_trial_numbers=top_k_trial_numbers,
        wall_time_seconds=wall_time,
        storage_path=storage_path,
        output_dir=output_dir if output_dir is not None else Path("."),
    )
