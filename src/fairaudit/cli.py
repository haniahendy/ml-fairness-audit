"""Command line entry point.

Reads a CSV with at least label, score-or-prediction, and group columns,
binarizes scores at a configurable threshold, and prints the audit.
"""

from __future__ import annotations

import argparse
import csv
import sys
from typing import List, Tuple

from .metrics import audit
from .report import render_json, render_text


def load_rows(
    path: str,
    label_col: str,
    score_col: str,
    group_col: str,
    threshold: float,
    scores_are_predictions: bool,
) -> List[Tuple[int, int, str]]:
    rows: List[Tuple[int, int, str]] = []
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        missing = {label_col, score_col, group_col} - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"missing columns in {path}: {sorted(missing)}")
        for line_no, record in enumerate(reader, start=2):
            try:
                label = int(float(record[label_col]))
                raw = float(record[score_col])
                group = record[group_col]
            except (TypeError, ValueError):
                raise SystemExit(f"bad numeric value on CSV line {line_no}")
            if label not in (0, 1):
                raise SystemExit(f"label must be 0/1 on CSV line {line_no}")
            prediction = int(raw) if scores_are_predictions else int(raw >= threshold)
            if prediction not in (0, 1):
                raise SystemExit(f"prediction must be 0/1 on CSV line {line_no}")
            rows.append((label, prediction, group))
    if not rows:
        raise SystemExit(f"no data rows in {path}")
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="fairaudit",
        description="Group fairness audit for a binary classifier's outputs.",
    )
    parser.add_argument("csv", help="CSV file with label, score, and group columns")
    parser.add_argument("--label", default="y_true", help="ground-truth column (0/1)")
    parser.add_argument("--score", default="y_score", help="score column in [0,1]")
    parser.add_argument("--group", default="group", help="group membership column")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="score threshold for a positive prediction")
    parser.add_argument("--predictions", action="store_true",
                        help="score column already holds 0/1 predictions")
    parser.add_argument("--reference", default=None,
                        help="reference group (default: the largest group)")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    args = parser.parse_args(argv)

    rows = load_rows(args.csv, args.label, args.score, args.group,
                     args.threshold, args.predictions)
    result = audit(rows, reference=args.reference)
    out = render_json(result) if args.json else render_text(result)
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
