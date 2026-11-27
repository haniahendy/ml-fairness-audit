"""Render audit results as JSON or a plain-text table."""

from __future__ import annotations

import json
from typing import Optional

from .metrics import AuditResult


def _fmt(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:+.4f}"


def render_json(result: AuditResult) -> str:
    payload = {
        "reference_group": result.reference,
        "groups": {
            name: {
                "n": gm.n,
                "base_rate": round(gm.base_rate, 6),
                "selection_rate": round(gm.selection_rate, 6),
                "tpr": None if gm.tpr is None else round(gm.tpr, 6),
                "fpr": None if gm.fpr is None else round(gm.fpr, 6),
                "precision": None if gm.precision is None else round(gm.precision, 6),
            }
            for name, gm in sorted(result.groups.items())
        },
        "gaps_vs_reference": {
            name: {
                "demographic_parity_diff": gap.demographic_parity_diff,
                "equal_opportunity_diff": gap.equal_opportunity_diff,
                "equalized_odds_max_abs_diff": gap.equalized_odds_max_abs_diff,
                "predictive_parity_diff": gap.predictive_parity_diff,
            }
            for name, gap in sorted(result.gaps.items())
        },
    }
    return json.dumps(payload, indent=2)


def render_text(result: AuditResult) -> str:
    lines = [
        f"reference group: {result.reference}",
        "",
        f"{'group':<16} {'n':>5} {'base':>7} {'sel':>7} {'tpr':>7} {'fpr':>7} {'prec':>7}",
    ]
    for name, gm in sorted(result.groups.items()):
        lines.append(
            f"{name:<16} {gm.n:>5} {gm.base_rate:>7.3f} {gm.selection_rate:>7.3f} "
            f"{_fmt(gm.tpr):>7} {_fmt(gm.fpr):>7} {_fmt(gm.precision):>7}"
        )
    lines += ["", f"{'group':<16} {'dp_diff':>9} {'eo_diff':>9} {'eodds':>9} {'pp_diff':>9}"]
    for name, gap in sorted(result.gaps.items()):
        if name == result.reference:
            continue
        lines.append(
            f"{name:<16} {_fmt(gap.demographic_parity_diff):>9} "
            f"{_fmt(gap.equal_opportunity_diff):>9} "
            f"{_fmt(gap.equalized_odds_max_abs_diff):>9} "
            f"{_fmt(gap.predictive_parity_diff):>9}"
        )
    return "\n".join(lines)
