"""Portfolio construction variants for the Larger Universe v2 study.

The v1 study used a single construction (rank top-30 equal-weight + caps);
v2 tests whether portfolio-construction changes improve regime consistency
without giving back the v1 model's excess return.

Each variant subclasses `ConstructionVariant` from `base.py` and implements
`construct(state) -> weights`. The backtest engine calls `construct()` at
each rebalance, threading per-rebalance context via `ConstructionState`.

Variants:
    BaselineVariant            — v1 reproduction (control)
    VolTargetVariant           — B1: 15% annualized vol target via gross scaling
    ConvictionWeightedVariant  — B2: softmax(score/T=0.5) within top-30
    DynamicTopNVariant         — B3: N varies 15-50 by score-dispersion percentile
    ConcentrationPenaltiesVariant — B4: persistence + sector-overweight penalties
    DefensiveSleevesVariant    — B5: 70/30 equity/defensive; 50/50 in stress
    SmallerCapsVariant         — B6: baseline with 4% individual cap (vs 7.5%)

Shared cap-enforcement logic lives in `caps.py` and matches v1's exact
behavior for the v2-baseline reproducibility check.

The dashboard discovers multi-variant studies via the presence of
`variant_meta.json` at the study root; each variant's artifacts live under
`<study>/<variant_subdir>/contract_v1/`. See `dashboard_contract_v1.md`.
"""
from src.equities.portfolio_construction.base import (
    ConstructionState,
    ConstructionVariant,
)
from src.equities.portfolio_construction.baseline import BaselineVariant
from src.equities.portfolio_construction.vol_target import VolTargetVariant
from src.equities.portfolio_construction.conviction_weighted import (
    ConvictionWeightedVariant,
)
from src.equities.portfolio_construction.dynamic_topn import DynamicTopNVariant
from src.equities.portfolio_construction.concentration_penalties import (
    ConcentrationPenaltiesVariant,
)
from src.equities.portfolio_construction.defensive_sleeves import (
    DefensiveSleevesVariant,
)
from src.equities.portfolio_construction.smaller_caps import SmallerCapsVariant

__all__ = [
    "ConstructionState",
    "ConstructionVariant",
    "BaselineVariant",
    "VolTargetVariant",
    "ConvictionWeightedVariant",
    "DynamicTopNVariant",
    "ConcentrationPenaltiesVariant",
    "DefensiveSleevesVariant",
    "SmallerCapsVariant",
]


def get_variant_by_name(name: str, **kwargs) -> ConstructionVariant:
    """Factory: instantiate a variant from its canonical name.

    Canonical names match `variant_meta.json.variants[].name`:
      - "baseline"
      - "b1_vol_target"
      - "b2_conviction_weighted"
      - "b3_dynamic_topn"
      - "b4_concentration_penalties"
      - "b5_defensive_sleeves"
      - "b6_smaller_caps"
    """
    registry = {
        "baseline": BaselineVariant,
        "b1_vol_target": VolTargetVariant,
        "b2_conviction_weighted": ConvictionWeightedVariant,
        "b3_dynamic_topn": DynamicTopNVariant,
        "b4_concentration_penalties": ConcentrationPenaltiesVariant,
        "b5_defensive_sleeves": DefensiveSleevesVariant,
        "b6_smaller_caps": SmallerCapsVariant,
    }
    if name not in registry:
        raise ValueError(
            f"Unknown variant name: {name!r}. "
            f"Valid names: {sorted(registry.keys())}"
        )
    return registry[name](**kwargs)
