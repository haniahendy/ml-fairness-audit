# ml-fairness-audit

Group fairness audit for a binary classifier's outputs. Reads a CSV of
ground-truth labels, model scores, and group membership, then reports the
standard pairwise fairness gaps against a reference group:

- **Demographic parity difference**: selection-rate gap
- **Equal opportunity difference**: true-positive-rate gap
- **Equalized odds**: max of |TPR gap| and |FPR gap|
- **Predictive parity difference**: precision gap

Pure Python 3.10+, zero dependencies; metrics are computed from first
principles on confusion counts, not wrapped from a framework.

## Usage

```bash
PYTHONPATH=src python3 -m fairaudit.cli predictions.csv \
    --label y_true --score y_score --group group --threshold 0.5
```

```
reference group: A

group                n    base     sel     tpr     fpr    prec
A                    8   0.500   0.500 +0.7500 +0.2500 +0.7500
B                    8   0.500   0.375 +0.2500 +0.5000 +0.3333

group              dp_diff   eo_diff     eodds   pp_diff
B                  -0.1250   -0.5000   +0.5000   -0.4167
```

`--json` emits the same numbers as JSON for downstream tooling;
`--predictions` treats the score column as already-binarized 0/1
predictions; `--reference` overrides the default (largest group).

## Input contract

CSV with a header row. Labels must be 0/1; scores are binarized at
`--threshold` (default 0.5). Any group column works (gender, age band,
region): the tool is agnostic to what the groups mean.

## Honest limits

- Binary outcomes and binary predictions only; no multi-class or
  regression support, and no intersectional (multi-attribute) breakdown , 
  run it once per attribute instead.
- Rate gaps are descriptive, not causal: a non-zero gap does not say why
  it exists, and a zero gap does not prove the model is fair.
- Groups with no positives (or no negatives) have undefined TPR (FPR);
  these render as `n/a` rather than being silently imputed.
- Small groups produce noisy rates; the tool reports counts so sample
  size is visible, but applies no significance testing.
- Threshold choice changes every gap; audit at the threshold you ship.

## Tests

```bash
PYTHONPATH=src python3 -m unittest discover -s tests
```

10 tests: hand-computed confusion counts and rates, undefined-rate
handling, reference selection and gap math, report rendering.
