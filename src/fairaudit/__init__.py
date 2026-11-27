"""Group fairness audit for binary classifiers.

Computes per-group performance and the standard pairwise fairness gaps
(demographic parity, equal opportunity, equalized odds, predictive
parity) from a CSV of ground-truth labels, scores, and group membership.
"""

from .metrics import GroupMetrics, audit, group_metrics
from .report import render_json, render_text

__all__ = [
    "GroupMetrics",
    "audit",
    "group_metrics",
    "render_json",
    "render_text",
]

__version__ = "0.1.0"
