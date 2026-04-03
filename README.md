# Thesis


## Fuzzy model

A fuzzy time series inference model that uses L lag days as inputs to predict the next demand value (t+1).

### Overview

**Training** converts historical demand into a rule base by:
1. Fuzzifying each demand value — assigning it to the fuzzy set with the highest membership degree (ties broken by lower index)
2. Building L-lagged input-output pairs from the training sequence
3. Grouping pairs that share the same antecedent fuzzy pattern and averaging their crisp output values into a single rule

**Prediction** uses all membership values (not just the dominant one) to fire rules:
- For each rule, firing strength is the product of the membership degrees of the rule's antecedent fuzzy sets evaluated on the input vector
- If no rules fire, the model cannot produce a prediction (a known limitation vs. regression models)
- If one or more rules fire, the prediction is the weighted average of their crisp outputs, weighted by firing strength

Functions:
- Train
- Predict

---

### Train

Input:
- Demand data
- Number of lags (L)
- Fuzzy sets

Output:
- Rule base: one rule per unique antecedent fuzzy pattern, each with a single crisp output (average of all matching training outputs)

Number of training input-output pairs:

```
N_pairs = N_train - L
```

#### Example

Input:

Number of lags L=3

Demand data

| Date       | Demand |
|------------|--------|
| 2026-01-01 | 2      |
| 2026-01-02 | 8      |
| 2026-01-03 | 10     |
| 2026-01-04 | 4      |
| 2026-01-05 | 8      |

Fuzzy sets

| Fuzzy set index | Fuzzy set name | Fuzzy set type | a | b  | c  | direction |
|-----------------|----------------|----------------|---|----|----|-----------|
| 0               | Low demand     | Left shoulder  | 1 | 5  | -  | left      |
| 1               | High demand    | Triangular     | 4 | 8  | 12 | -         |
| 2               | Very high      | Right shoulder | 6 | 12 | -  | right     |

Step 1 — Fuzzify all demand values:

| Date       | Demand | Dominant fuzzy set |
|------------|--------|--------------------|
| 2026-01-01 | 2      | Low demand         |
| 2026-01-02 | 8      | High demand        |
| 2026-01-03 | 10     | Very high          |
| 2026-01-04 | 4      | Low demand         |
| 2026-01-05 | 8      | High demand        |

Step 2 — Build lagged input-output pairs (N_pairs = 5 - 3 = 2):

| Fuzzy lag 1 | Fuzzy lag 2 | Fuzzy lag 3 | Actual demand (y) |
|-------------|-------------|-------------|-------------------|
| Low demand  | High demand | Very high   | 4                 |
| High demand | Very high   | Low demand  | 8                 |

Step 3 — Group by antecedent pattern and average outputs (no duplicates here, so rule base = pairs):

Output rule base:

| Fuzzy lag 1 | Fuzzy lag 2 | Fuzzy lag 3 | Rule output (y) |
|-------------|-------------|-------------|-----------------|
| Low demand  | High demand | Very high   | 4               |
| High demand | Very high   | Low demand  | 8               |

---

### Predict

Input:
- Starting demand data (last L real demand values)
- Horizon

Output:
- Prediction data

For each prediction step:
1. Compute all membership degrees for each input value across all fuzzy sets
2. For each rule, compute firing strength: product of membership degrees of the rule's antecedent fuzzy sets for the corresponding input lags
3. Discard rules with firing strength = 0
4. Prediction = weighted average of fired rule outputs:

```
Y = sum(w_i * y_i) / sum(w_i)
```

Where w_i is the firing strength of rule i and y_i is its crisp output.

For horizons > 1, the model feeds its own predictions back as inputs (recursive pipeline). This causes compounding errors as the horizon grows.

Infer the number of lags from the rule base.

#### Example

Input:

Horizon = 1

Starting demand data

| Date       | Demand |
|------------|--------|
| 2026-02-01 | 3      |
| 2026-02-02 | 6      |
| 2026-02-03 | 9      |

Rule base (from training):

| Fuzzy lag 1 | Fuzzy lag 2 | Fuzzy lag 3 | Rule output (y) |
|-------------|-------------|-------------|-----------------|
| Low demand  | High demand | Very high   | 4               |
| High demand | Very high   | Low demand  | 8               |

Membership degrees for input [3, 6, 9]:

| Input value | Low demand | High demand | Very high |
|-------------|------------|-------------|-----------|
| 3 (lag 1)   | 0.5        | 0.0         | 0.0       |
| 6 (lag 2)   | 0.0        | 0.5         | 0.0       |
| 9 (lag 3)   | 0.0        | 0.25        | 0.75      |

Firing strengths:
- Rule 1 (Low, High, Very high): 0.5 * 0.5 * 0.75 = 0.1875
- Rule 2 (High, Very high, Low): 0.0 * ... = 0 → not fired

Prediction: (4 * 0.1875) / 0.1875 = **4**

Output:

| Date       | Prediction |
|------------|------------|
| 2026-02-04 | 4          |

---

## Evaluation

Based on the scoring method by Nicholas Vandeput.
Combines mean absolute error (MAE) and bias. Lower score is better.

Calculation:
1. error_i = prediction_i - actual_demand_i
2. abs_error_i = ABS(error_i)
3. score = SUM(error_i) + SUM(abs_error_i)

Two evaluation setups:

**Primary — Rolling 1-step-ahead (walk-forward):** At each test day, use the last L real demand values to predict t+1. Move forward one day, repeat across all test days. This is directly comparable to the base model.

**Secondary — Recursive multi-step pipeline:** Starting from the first test day, feed predictions back as inputs for subsequent steps. Shows how error compounds over the forecast horizon.

Input:
- Base model predictions
- Fuzzy model predictions
- Actual demand

Output:
- Score per model

### Example

Base model predictions

| Date       | Prediction |
|------------|------------|
| 2026-02-04 | 7.5        |

Fuzzy model predictions

| Date       | Prediction |
|------------|------------|
| 2026-02-04 | 9          |

Actual demand

| Date       | Demand |
|------------|--------|
| 2026-02-04 | 8      |

Output:

| Base model score | Fuzzy model score |
|------------------|-------------------|
| 1                | 2                 |

---

## Base model — 7-day moving average

The t+1 prediction is the average of the previous 7 days of real demand values. Used as the benchmark for the fuzzy model evaluation.
