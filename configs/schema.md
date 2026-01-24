# QUANT-NEURAL Config Schema (v2026.1)

## project

- seed (int): global seed for numpy/sklearn/tf
- as_of_date (YYYY-MM-DD): point-in-time cutoff

## preprocessing.rankgauss

- enabled (bool)
- n_quantiles (int): bounded by n_train; QuantileTransformer can overfit on very small datasets (use min(n_quantiles, n_train))

## preprocessing.hp_filter

- enabled (bool)
- freq (enum: M|Q|Y)
- mode (enum: classic|ravn_uhlig|manual)
  - classic: M=14400, Q=1600, Y=100
  - ravn_uhlig: M=129600, Q=1600, Y=6.25 (per statsmodels doc)
- lamb_manual (float|null): required if mode=manual

## preprocessing.hamilton_filter

- enabled (bool)
- h (int), p (int): regression horizon and lag count
- macro_lag_months (int): MUST be >= h if Hamilton output is used in live trading logic

## factors.winsorize

- lower_q (float), upper_q (float): default 0.01 / 0.99

## selection.lasso

- n_splits (int): TimeSeriesSplit folds
- max_iter (int)

## regime

- threshold (float): action rule threshold
- C (float), max_iter (int), class_weight (null|"balanced")
- calibration.enabled (bool): enables probability calibration (Phase 5.2)
- calibration.method ("temperature")

## models

- type ("mlp"|"kan")
- mlp.\*: training params

## conformal

- enabled (bool)
- alpha (float): e.g., 0.05 for 95% intervals
- method ("mapie_split")
