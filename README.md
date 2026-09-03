# ML Learning

[![Tests](https://github.com/KutayStudy/ml-learning/actions/workflows/tests.yml/badge.svg)](https://github.com/KutayStudy/ml-learning/actions/workflows/tests.yml)


A lightweight, notebook-first repository for learning the mathematical and practical foundations of machine learning.

The repository focuses on runnable experiments, small exercises, and practical learning artifacts rather than extensive standalone notes.

## Prep Summary

Foundational phase before the 6-month roadmap begins: environment and repo setup, SVD/PCA, MLE and MAP (worked by hand on paper), NumPy (fundamentals through broadcasting and linear algebra), pandas (fundamentals through merging, cleaning, and MultiIndex), and SQL analytics with DuckDB. All exercise and practice notebooks live under `notebooks/prep/`.

## Month 1 — Flagship Project (in progress)

Started the real roadmap. Dataset: Duolingo's Learning Traces — the data behind their published half-life regression research. Target is `p_recall`, probability of recalling a word in a session. Sampled 2,500 users (all their sessions, not random rows) so I can group-split by user later — 16,382 rows total.

Data dictionary turned up something important: `p_recall` is literally `session_correct / session_seen`. Can't use those two as features or the model just reads off the answer. Target's also skewed hard toward 1.0, only 29 distinct values despite being a float.

Added `history_accuracy` and a log-transformed `lag_days` (skewed, used `log1p` for the zeros). First cleaning pass done too — fixed timestamps, made the low-cardinality columns categorical, checked for impossible values, stripped whitespace on the word columns.

`grammar_tags` is ~10% missing, but it's MAR — it tracks part of speech, since non-inflecting word classes just don't have tags — so it gets a `no_gram` sentinel instead of an imputed value. EDA on the numeric columns showed `history_seen`/`history_correct` extremely right-skewed (a few users drill one word hundreds of times), `lag_days` moderately skewed, and `p_recall` clumped so hard at 1.0 that modified z-score divides by zero — IQR is the only outlier method that holds up. Feature–target correlations are all weak (|r| < 0.12), so whatever signal exists is in combinations, not single columns.

Ran a leakage audit: `p_recall` is a direct function of `session_correct`/`session_seen` (permanently excluded), `user_id` repeats ~6.5×/user so the split has to be `GroupKFold` on user, and a temporal check on repeated (user, lexeme) sessions confirmed `history_*` only ever increases. Held out 15% of users (375 users, 1,944 rows) as an untouched final test set, kept the rest (2,125 users, 14,438 rows) as a CV pool. Baselines there: mean gets RMSE 0.276, median gets MAE 0.106 (it matches the >50% of rows sitting exactly at 1.0). Chose RMSE as the primary metric — overpredicting recall is the failure that matters. A default `RandomForestRegressor` scored RMSE 0.155 in-sample, enough to confirm real signal; proper CV evaluation is Month 2.

The split lives in `data/split_users.csv` and is the one data artifact kept under version control. It was originally re-derived inline in each notebook by shuffling `df["user_id"].unique()` — which returns users in *row order*, so the SQL join below reordered the rows and the same seed quietly produced a different hold-out set (70 of 375 users in common). Users are now sorted before shuffling and the split is written once in part 11; because it's keyed on `user_id`, it applies unchanged to every dataset version.

Added a `difficulty_rank_in_language` feature through a SQL `LEFT JOIN` on `lexeme_id` (window functions and joins were the new SQL here). All of it — cleaning, the missing-value fill, both engineered features, and the difficulty join — is wired into a `typer` CLI (`src/data/build.py`) that turns the sampled dataset into the model-ready table in one command, with a `pandera` schema that fails loud on out-of-range values:

```bash
python -m src.data.build data/duolingo_flagship.csv data/duolingo_flagship_v5.csv
```

The one step deliberately left outside the pipeline is part 2's user sampling: it used an unseeded `ORDER BY random()`, so re-running it would pick a different 2,500 users and invalidate the saved split. That sample is frozen as the pipeline's input instead. The pipeline functions are covered by `pytest`, and a GitHub Actions workflow runs the suite on every push. Findings so far are collected in [`reports/eda.md`](reports/eda.md).

Also started Andrew Ng's supervised-learning material and rebuilt Week 1 from scratch — the linear model, squared-error cost, and gradient descent — plus a separate note deriving why squared error follows from a Gaussian-noise assumption through maximum likelihood. Picked MSE as the first modeling objective for the flagship, with a plan to inspect the residual distribution before committing to it.

## Mathematical Foundations

- **SVD and PCA:** Singular vectors, singular values, low-rank approximation, explained variance, and dimensionality reduction.
- **Maximum Likelihood Estimation:** Choosing parameters that maximize the likelihood of the observed data.
- **MAP and Regularization:** Combining likelihood with prior knowledge; Gaussian priors lead to L2 regularization and Ridge regression.

## Machine Learning Foundations

- **Linear Regression:** The one-variable model $f_{w,b}(x) = wx + b$, the distinction between data ($x$) and parameters learned from data ($w, b$), and the geometric roles of slope and intercept.
- **Squared-Error Cost:** Residuals aggregated into a single scalar $J(w,b)$, framed as the objective to minimize, and the convexity of the cost surface for this problem.
- **Gradient Descent:** Partial derivatives of the cost, simultaneous update of $w$ and $b$, the learning rate $\alpha$ as step size, and recognizing convergence versus divergence — implemented from scratch on a toy dataset.
- **MLE to Squared Loss:** Assuming independent, zero-mean, constant-variance Gaussian residual noise makes maximum likelihood estimation for linear regression equivalent to minimizing squared error; the loss comes from a modeling assumption, while gradient descent is only the optimizer.

## Numerical Computing

- **NumPy Fundamentals:** Array creation, data types, indexing, slicing, Boolean masking, fancy indexing, reshaping, flattening, transposition, and vectorized operations.
- **Broadcasting and Reduction:** Broadcasting rules, axis-based reduction (`sum`, `mean`, `std`, `argmin`/`argmax`), and `keepdims` for re-broadcasting after a reduction.
- **Linear Algebra Operations:** Dot products, matrix multiplication, norms (with safe handling of zero vectors), solving linear systems (`solve` vs `inv`), determinants, matrix rank, eigendecomposition (`eig` vs `eigh`), and least squares (`lstsq`).
- **NumPy Practice:** Completed exercises 1–60 from the [numpy-100](https://github.com/rougier/numpy-100) collection.

## Data Manipulation (pandas)

- **pandas Fundamentals:** `Series` and `DataFrame` basics, `.loc`/`.iloc`, boolean filtering (`&`, `|`, `~`, `.isin()`, `.query()`), `sort_values`, and loading data with `read_csv` (`dtype`, `parse_dates`).
- **Grouping and Aggregation:** `groupby` with multiple aggregation functions, named aggregation, `as_index`, and the distinction between `size` and `count`.
- **Advanced pandas:** Merging and joining (`validate`, `indicator`), string cleaning with regex, datetime handling and resampling, missing-value handling, MultiIndex, and categorical dtype.
- **Practice:** Targeted subset of exercises from [100-pandas-puzzles](https://github.com/ajcr/100-pandas-puzzles) (not full completion — selected for relevance to real tabular-data work).

## SQL Analytics (DuckDB)

- **SQL over DataFrames:** Querying pandas DataFrames directly with DuckDB without creating database files.
- **Aggregation:** Customer-level analysis using `GROUP BY`, `SUM`, and `AVG`.
- **Window Functions:** Ranking and sequential analysis with `ROW_NUMBER`, `RANK`, `DENSE_RANK`, `LAG`, and `LEAD`.
- **Window Frames:** Running totals and moving averages with `ROWS BETWEEN`.
- **Analytical Filtering:** Filtering window-function results with `QUALIFY` and common table expressions.
- **Relational Operations:** Joining sales and customer data with SQL joins.
- **Time-Series Aggregation:** Monthly sales analysis using `DATE_TRUNC`, compared with pandas `resample`.

## Notebooks

Prep exercises live under `notebooks/prep/`, flagship work under `notebooks/month1/`.

Prep:

- [`s2_svd_pca.ipynb`](notebooks/prep/s2_svd_pca.ipynb) — SVD, low-rank approximation, and PCA
- [`S5_NumPy_Examples.ipynb`](notebooks/prep/S5_NumPy_Examples.ipynb) — NumPy fundamentals with runnable examples
- [`100_Numpy_exercises.ipynb`](notebooks/prep/100_Numpy_exercises.ipynb) — Solutions to numpy-100 exercises 1–60
- [`S6_NumPy_Broadcasting_Axis_Linalg.ipynb`](notebooks/prep/S6_NumPy_Broadcasting_Axis_Linalg.ipynb) — Broadcasting, axis-based reduction, and linear algebra operations
- [`S7_Pandas_Fundamentals_and_Exercises.ipynb`](notebooks/prep/S7_Pandas_Fundamentals_and_Exercises.ipynb) — pandas fundamentals with a self-built dataset
- [`S8_Pandas_Advanced_and_Exercises.ipynb`](notebooks/prep/S8_Pandas_Advanced_and_Exercises.ipynb) — Merge/join, string cleaning, dates, missing values, MultiIndex, and categorical dtype
- [`100-pandas-puzzles.ipynb`](notebooks/prep/100-pandas-puzzles.ipynb) — Targeted practice from 100-pandas-puzzles
- [`S9_SQL.ipynb`](notebooks/prep/S9_SQL.ipynb) — DuckDB aggregation, window functions, joins, and monthly sales analysis

Month 1 (flagship):

- [`part2_dataset_selection.ipynb`](notebooks/month1/part2_dataset_selection.ipynb) — sampling and loading the flagship dataset
- [`part3_data_dictionary.ipynb`](notebooks/month1/part3_data_dictionary.ipynb) — column types, missingness, cardinality, target analysis
- [`part4_numpy_features.ipynb`](notebooks/month1/part4_numpy_features.ipynb) — engineered features (accuracy rate, log-transformed lag)
- [`part5_cleaning_v1.ipynb`](notebooks/month1/part5_cleaning_v1.ipynb) — first cleaning pass (types, whitespace, validity checks)
- [`part6_missing_data.ipynb`](notebooks/month1/part6_missing_data.ipynb) — missing-data mechanisms and the impute-before-split leakage trap
- [`part8_eda_distributions.ipynb`](notebooks/month1/part8_eda_distributions.ipynb) — distribution shapes and IQR / z-score / modified z-score outlier checks
- [`part9_eda_correlation.ipynb`](notebooks/month1/part9_eda_correlation.ipynb) — Pearson vs Spearman, categorical breakdowns, leakage screening
- [`part10_leakage.ipynb`](notebooks/month1/part10_leakage.ipynb) — target, contamination, temporal, and group leakage on toy data and the flagship
- [`part11_split.ipynb`](notebooks/month1/part11_split.ipynb) — KFold vs TimeSeriesSplit vs GroupKFold; group split on `user_id`
- [`part12_baseline.ipynb`](notebooks/month1/part12_baseline.ipynb) — Dummy regressors and the MAE / RMSE / R² trade-off
- [`part13_firstModel.ipynb`](notebooks/month1/part13_firstModel.ipynb) — default RandomForest as a signal check against the baseline
- [`part14_sqlFeature.ipynb`](notebooks/month1/part14_sqlFeature.ipynb) — SQL window functions and joins for a word-difficulty feature
- [`part16_pipeline&SchemaValidation.ipynb`](notebooks/month1/part16_pipeline&SchemaValidation.ipynb) — CLI pipeline with pandera schema validation
- [`part21_linear_regression_foundations.ipynb`](notebooks/month1/part21_linear_regression_foundations.ipynb) — linear model, squared-error cost, and from-scratch gradient descent
- [`part22_mle_to_squared_loss.md`](notebooks/month1/part22_mle_to_squared_loss.md) — deriving the squared-error loss from Gaussian-noise MLE

Supporting flagship artifacts: [`notes/`](notes/) (problem statement, data dictionary, missing-value plan), [`reports/eda.md`](reports/eda.md), the [`src/data/build.py`](src/data/build.py) CLI pipeline, and [`tests/`](tests/) with the GitHub Actions workflow in [`.github/workflows/tests.yml`](.github/workflows/tests.yml).

## Repository Structure

```
ml-learning/
├── notebooks/
│   ├── prep/
│   │   ├── 100_Numpy_exercises.ipynb
│   │   ├── 100-pandas-puzzles.ipynb
│   │   ├── s2_svd_pca.ipynb
│   │   ├── S5_NumPy_Examples.ipynb
│   │   ├── S6_NumPy_Broadcasting_Axis_Linalg.ipynb
│   │   ├── S7_Pandas_Fundamentals_and_Exercises.ipynb
│   │   ├── S8_Pandas_Advanced_and_Exercises.ipynb
│   │   └── S9_SQL.ipynb
│   └── month1/
│       ├── part2_dataset_selection.ipynb
│       ├── part3_data_dictionary.ipynb
│       ├── part4_numpy_features.ipynb
│       ├── part5_cleaning_v1.ipynb
│       ├── part6_missing_data.ipynb
│       ├── part8_eda_distributions.ipynb
│       ├── part9_eda_correlation.ipynb
│       ├── part10_leakage.ipynb
│       ├── part11_split.ipynb
│       ├── part12_baseline.ipynb
│       ├── part13_firstModel.ipynb
│       ├── part14_sqlFeature.ipynb
│       ├── part16_pipeline&SchemaValidation.ipynb
│       ├── part21_linear_regression_foundations.ipynb
│       └── part22_mle_to_squared_loss.md
├── notes/
│   ├── data_dictionary.md
│   ├── missing_value_plan.md
│   └── problem_statement.md
├── reports/
│   ├── eda.md
│   └── figures/
├── src/
│   └── data/
│       └── build.py
├── tests/
│   └── test_build.py
├── .github/
│   └── workflows/
│       └── tests.yml
├── data/
│   └── split_users.csv        # the only data file kept in git
├── pyproject.toml
├── uv.lock
└── README.md
```
