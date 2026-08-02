# ML Learning

A lightweight, notebook-first repository for learning the mathematical and practical foundations of machine learning.

The repository focuses on runnable experiments, small exercises, and practical learning artifacts rather than extensive standalone notes.

## Prep Summary

Foundational phase before the 6-month roadmap begins: environment and repo setup, SVD/PCA, MLE and MAP (worked by hand on paper), NumPy (fundamentals through broadcasting and linear algebra), pandas (fundamentals through merging, cleaning, and MultiIndex), and SQL analytics with DuckDB. All exercise and practice notebooks live under `notebooks/prep/`.

## Mathematical Foundations

- **SVD and PCA:** Singular vectors, singular values, low-rank approximation, explained variance, and dimensionality reduction.
- **Maximum Likelihood Estimation:** Choosing parameters that maximize the likelihood of the observed data.
- **MAP and Regularization:** Combining likelihood with prior knowledge; Gaussian priors lead to L2 regularization and Ridge regression.

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

All located under `notebooks/prep/`:

- [`s2_svd_pca.ipynb`](notebooks/prep/s2_svd_pca.ipynb) — SVD, low-rank approximation, and PCA
- [`S5_NumPy_Examples.ipynb`](notebooks/prep/S5_NumPy_Examples.ipynb) — NumPy fundamentals with runnable examples
- [`100_Numpy_exercises.ipynb`](notebooks/prep/100_Numpy_exercises.ipynb) — Solutions to numpy-100 exercises 1–60
- [`S6_NumPy_Broadcasting_Axis_Linalg.ipynb`](notebooks/prep/S6_NumPy_Broadcasting_Axis_Linalg.ipynb) — Broadcasting, axis-based reduction, and linear algebra operations
- [`S7_Pandas_Fundamentals_and_Exercises.ipynb`](notebooks/prep/S7_Pandas_Fundamentals_and_Exercises.ipynb) — pandas fundamentals with a self-built dataset
- [`S8_Pandas_Advanced_and_Exercises.ipynb`](notebooks/prep/S8_Pandas_Advanced_and_Exercises.ipynb) — Merge/join, string cleaning, dates, missing values, MultiIndex, and categorical dtype
- [`100-pandas-puzzles.ipynb`](notebooks/prep/100-pandas-puzzles.ipynb) — Targeted practice from 100-pandas-puzzles
- [`S9_SQL.ipynb`](notebooks/prep/S9_SQL.ipynb) — DuckDB aggregation, window functions, joins, and monthly sales analysis

## Repository Structure

```
ml-learning/
├── notebooks/
│   └── prep/
│       ├── 100_Numpy_exercises.ipynb
│       ├── 100-pandas-puzzles.ipynb
│       ├── s2_svd_pca.ipynb
│       ├── S5_NumPy_Examples.ipynb
│       ├── S6_NumPy_Broadcasting_Axis_Linalg.ipynb
│       ├── S7_Pandas_Fundamentals_and_Exercises.ipynb
│       ├── S8_Pandas_Advanced_and_Exercises.ipynb
│       └── S9_SQL.ipynb
├── data/
├── pyproject.toml
├── uv.lock
└── README.md
```