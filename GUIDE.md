# Reviewer's Guide — Fuzzy Time-Series Demand Forecasting

This document describes the mathematical methodology, design decisions, and
experimental setup of the thesis project.  Code navigation pointers are
included so the reviewer can locate each component without running anything.

---

## 1. Problem Statement

Given a univariate daily demand time series x_1, …, x_N, predict x_{t+1} using
the L most recent observations as input.  The fuzzy model is benchmarked against
a 7-day simple moving average.

---

## 2. Data and Preprocessing

**Source:** `data/Szakdoga_adat.csv` — 789 daily observations (2024-01-01 to
2026-02-27) with columns: date, sales, stock.

**Stockout imputation** (`utilities/utils.py → get_prepared_demand_df`):
On days where stock = 0, recorded sales are a censored observation of true demand.
These are treated as missing (NaN) and replaced by linear interpolation between
the last in-stock and first post-stockout demand values.  This is a pragmatic
choice given that actual unfulfilled demand is unobserved.

**Holiday removal:** Hungarian statutory public holidays are excluded from the
test set before evaluation (`utilities/utils.py → get_holidays_df`).  Holiday
demand is systematically atypical and would inflate error metrics independently
of model quality.

**Demand characterisation** (`utilities/utils.py → calculate_adi, calculate_cv`):

| Metric | Value | Interpretation |
|--------|-------|----------------|
| ADI    | ≈ 1.0 | Demand occurs every period (continuous) |
| CV     | ≈ 0.59 | Moderate variability (CV² ≈ 0.35 < 0.49) |

Per the Syntetos-Boylan classification matrix, the series falls in the "smooth"
quadrant — appropriate for standard time-series methods.

---

## 3. Fuzzy Partition

**Code:** `data/Main.ipynb` (cells 6–7), `models/fuzzy.py`

Five linguistic sets partition the demand domain.  Boundaries are set at
approximate quartiles of the training distribution (rounded for interpretability):

| Set name       | Type          | a   | b   | c   |
|----------------|---------------|-----|-----|-----|
| VeryLowDemand  | Left shoulder | 0   | 25  | —   |
| LowDemand      | Triangular    | 0   | 25  | 50  |
| MediumDemand   | Triangular    | 25  | 50  | 100 |
| HighDemand     | Triangular    | 50  | 100 | 500 |
| VeryHighDemand | Right shoulder | 100 | 500 | —  |

**Shoulder sets at the boundaries** prevent extreme demand values from falling
outside the support of all sets (which would yield zero membership everywhere
and make fuzzification undefined).  A left shoulder ensures μ = 1 for all
x ≤ 0, a right shoulder ensures μ = 1 for all x ≥ 500.

Membership function implementations: `TriangularFuzzySet`, `ShoulderFuzzySet`
in `models/fuzzy.py`.

---

## 4. Rule Base Construction (Training)

**Code:** `models/fuzzy.py → create_rule_base_df`

**Algorithm (Chen, 1996; building on Song & Chissom, 1993):**

1. For each training observation x_t, create L lagged inputs:
   (x_{t-1}, x_{t-2}, …, x_{t-L}).

2. Fuzzify each lag by argmax membership — assign it to the set A_i with the
   highest μ_i(x):
   F(x) = argmax_i μ_i(x)
   This maps each lag to one of the five linguistic labels.

3. Form antecedent-consequent pairs:
   (F(x_{t-1}), F(x_{t-2}), …, F(x_{t-L})) → x_t

4. Group all pairs sharing the same L-tuple of set labels and average their
   consequents.  Each unique L-tuple becomes one rule with a single crisp output:
   y_r = mean{ x_t : antecedent of x_t = r }

**Result:** 418 unique rules for L = 7.

**Fuzzification choice:** The "all" strategy (`get_one_lag_all_fuzzification`)
was explored as an alternative — it assigns each lag value to every set with
μ > 0 (possibly duplicating rows), producing a richer but more redundant rule
base.  The "highest" (argmax) strategy is used in the final model.

---

## 5. Inference (Prediction)

**Code:** `models/fuzzy.py → fuzzy_forecast`, `fuzzy_forecast_pipeline`

Given input vector **x** = (x_{t}, x_{t-1}, …, x_{t-L+1}):

**Step 1 — Full fuzzification:**
Compute μ_i(x_j) for all sets A_i and all lags j = 1…L.  Unlike training
(which uses argmax), inference uses the continuous membership degree of
whichever set the rule prescribes for that lag.

**Step 2 — Firing strength (Larsen product t-norm):**
For rule r with antecedent pattern (A_{1}^r, A_{2}^r, …, A_{L}^r):

    w_r = ∏_{j=1}^{L} μ_{A_j^r}(x_j)

The algebraic product is used as the t-norm (conjunction operator).  This is
the Larsen product inference method (Larsen, 1980) — distinct from the Mamdani
system, which uses the minimum t-norm.  A rule fires only when every antecedent
membership degree is positive.

**Step 3 — Defuzzification (zero-order Sugeno weighted average):**

    ŷ = Σ_r (w_r · y_r) / Σ_r w_r

Each consequent y_r is a crisp scalar (the averaged training output for rule r),
so the output is the firing-strength-weighted average of constants.  This is
zero-order Sugeno (Takagi & Sugeno, 1985) defuzzification — not centre-of-gravity,
which applies to Mamdani systems with continuous fuzzy output sets.

**Fallback:** If Σ_r w_r = 0 (no rule fires), the prediction defaults to the
global mean of all rule consequents.  This occurred in 13 of 159 test days
(8.18%), most of which coincided with an unusual demand period in January 2026.

**Code path:**
`prepare_fuzzified_forecast_demand_df` constructs the cross-join of the input
vector with the rule base so that firing strengths can be computed in a single
vectorised operation.

---

## 6. Baseline Model

**Code:** `models/moving_average.py → forecast_moving_average`

    ŷ_{t+1} = (1/L) Σ_{k=1}^{L} x_{t-k+1}

Unweighted L-day moving average.  L = 7 throughout, matching the lag order
of the fuzzy model.

---

## 7. Evaluation Protocol

**Code:** `forecast/forecast.py → forecast_over_horizon`,
`evaluate/evaluate_model.py → get_model_score`

**Walk-forward (expanding-window) validation:**
At each test step t, only observations strictly prior to t are used to produce
the prediction.  The rule base is fixed (trained once on the training set);
only the input window advances.  This avoids look-ahead bias.

**Split:** 80% training / 20% test (chrono-ordered).  The first L days of the
test window serve as the initial input and are not evaluated.

**Score (Vandeput, 2021):**

    e_t = ŷ_t - x_t           (positive = over-forecast)
    MAE = mean(|e_t|)
    Bias = mean(e_t)
    MAE% = MAE / mean(x_t)
    Bias% = Bias / mean(x_t)
    Score = MAE% + |Bias%|

The score combines magnitude of error (MAE%) with systematic directional
skew (|Bias%|).  A model with low MAE but consistent over- or under-forecasting
is penalised more than one with the same MAE and zero bias.

---

## 8. Results

| Model          | Score | MAE   | MAE%  | Bias  | Bias% |
|----------------|-------|-------|-------|-------|-------|
| Fuzzy (L=7)    | 0.347 | 21.48 | 33.5% | -0.76 | -1.2% |
| Moving average | 0.414 | 26.27 | 40.9% | -0.31 | -0.5% |

The fuzzy model reduces the combined score by ~16% relative to the baseline.
Both models exhibit a slight negative bias (tendency to under-forecast), more
pronounced in the fuzzy model.

---

## 9. Parameter Selection

**Code:** `data/ParameterTuning.ipynb`

A reduced 3-set partition (VeryLow, Medium, VeryHigh) was used to efficiently
sweep L ∈ {1, …, 29}.  The minimum fuzzy score was achieved at L = 7 (score =
0.3388), which was then confirmed with the full 5-set partition in the main
experiment.

Results are in `data/experiment_runs.csv`.

---

## 10. Code Map

| Concept                        | File                              | Entry point                        |
|-------------------------------|-----------------------------------|------------------------------------|
| Membership functions           | `models/fuzzy.py`                 | `TriangularFuzzySet`, `ShoulderFuzzySet` |
| Rule base construction         | `models/fuzzy.py`                 | `create_rule_base_df`              |
| Inference / defuzzification    | `models/fuzzy.py`                 | `fuzzy_forecast`                   |
| Fuzzified input preparation    | `models/fuzzy.py`                 | `prepare_fuzzified_forecast_demand_df` |
| Baseline model                 | `models/moving_average.py`        | `forecast_moving_average`          |
| Stockout imputation            | `utilities/utils.py`              | `get_prepared_demand_df`           |
| Walk-forward loop              | `forecast/forecast.py`            | `forecast_over_horizon`            |
| Full experiment runner         | `forecast/forecast.py`            | `run_experiment_fuzzy_and_average_model` |
| Vandeput score                 | `evaluate/evaluate_model.py`      | `get_model_score`                  |
| Main experiment (full output)  | `data/Main.ipynb`                 | —                                  |
| Parameter sweep                | `data/ParameterTuning.ipynb`      | —                                  |
