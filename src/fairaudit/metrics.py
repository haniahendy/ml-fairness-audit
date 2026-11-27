"""Confusion-matrix derived fairness metrics.

Everything is computed from first principles on (label, prediction, group)
triples; there is no dependency on a modelling framework. The metrics are
defined for binary outcomes with the positive class equal to 1.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass
class GroupMetrics:
    """Confusion counts and derived rates for one group."""

    group: str
    n: int = 0
    positives: int = 0
    negatives: int = 0
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0

    @property
    def base_rate(self) -> float:
        return self.positives / self.n if self.n else 0.0

    @property
    def selection_rate(self) -> float:
        """Fraction predicted positive (demographic parity numerator)."""
        return (self.tp + self.fp) / self.n if self.n else 0.0

    @property
    def tpr(self) -> Optional[float]:
        """True positive rate; None when the group has no positives."""
        return self.tp / self.positives if self.positives else None

    @property
    def fpr(self) -> Optional[float]:
        """False positive rate; None when the group has no negatives."""
        return self.fp / self.negatives if self.negatives else None

    @property
    def precision(self) -> Optional[float]:
        """Positive predictive value; None when nothing predicted positive."""
        predicted_positive = self.tp + self.fp
        return self.tp / predicted_positive if predicted_positive else None


def group_metrics(rows: Iterable[Tuple[int, int, str]]) -> Dict[str, GroupMetrics]:
    """Aggregate (label, prediction, group) triples per group."""
    groups: Dict[str, GroupMetrics] = {}
    for label, prediction, group in rows:
        gm = groups.setdefault(group, GroupMetrics(group=group))
        gm.n += 1
        if label == 1:
            gm.positives += 1
        else:
            gm.negatives += 1
        if prediction == 1 and label == 1:
            gm.tp += 1
        elif prediction == 1 and label == 0:
            gm.fp += 1
        elif prediction == 0 and label == 0:
            gm.tn += 1
        else:
            gm.fn += 1
    return groups


def _diff(value: Optional[float], reference: Optional[float]) -> Optional[float]:
    if value is None or reference is None:
        return None
    return value - reference


@dataclass
class GroupGap:
    """Fairness gaps of one group against the reference group."""

    group: str
    demographic_parity_diff: Optional[float]
    equal_opportunity_diff: Optional[float]
    equalized_odds_max_abs_diff: Optional[float]
    predictive_parity_diff: Optional[float]


@dataclass
class AuditResult:
    reference: str
    groups: Dict[str, GroupMetrics]
    gaps: Dict[str, GroupGap] = field(default_factory=dict)


def audit(
    rows: Sequence[Tuple[int, int, str]],
    reference: Optional[str] = None,
) -> AuditResult:
    """Compute per-group metrics and gaps against the reference group.

    The reference defaults to the largest group. Rate differences are
    (group rate - reference rate), so a negative demographic parity diff
    means the group is selected *less* often than the reference.
    """
    groups = group_metrics(rows)
    if not groups:
        raise ValueError("no rows to audit")
    if reference is None:
        reference = max(groups.values(), key=lambda g: g.n).group
    if reference not in groups:
        raise ValueError(f"reference group {reference!r} not present in data")
    ref = groups[reference]
    gaps: Dict[str, GroupGap] = {}
    for name, gm in groups.items():
        tpr_d = _diff(gm.tpr, ref.tpr)
        fpr_d = _diff(gm.fpr, ref.fpr)
        eo_components = [abs(d) for d in (tpr_d, fpr_d) if d is not None]
        gaps[name] = GroupGap(
            group=name,
            demographic_parity_diff=_diff(gm.selection_rate, ref.selection_rate),
            equal_opportunity_diff=tpr_d,
            equalized_odds_max_abs_diff=max(eo_components) if eo_components else None,
            predictive_parity_diff=_diff(gm.precision, ref.precision),
        )
    return AuditResult(reference=reference, groups=groups, gaps=gaps)
