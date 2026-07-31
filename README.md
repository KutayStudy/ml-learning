# ML Learning

A lightweight, notebook-first repository for learning the mathematical and practical foundations of machine learning.

The repository focuses on runnable experiments, small exercises, and practical learning artifacts rather than extensive standalone notes.

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

## Notebooks

- [`s2_svd_pca.ipynb`](notebooks/s2_svd_pca.ipynb) — SVD, low-rank approximation, and PCA
- [`S5_NumPy_Examples.ipynb`](notebooks/S5_NumPy_Examples.ipynb) — NumPy fundamentals with runnable examples
- [`100_Numpy_exercises.ipynb`](notebooks/100_Numpy_exercises.ipynb) — Solutions to numpy-100 exercises 1–60
- [`S6_NumPy_Broadcasting_Axis_Linalg.ipynb`](notebooks/S6_NumPy_Broadcasting_Axis_Linalg.ipynb) — Broadcasting, axis-based reduction, and linear algebra operations
- [`s7_pandas_fundamentals.ipynb`](notebooks/s7_pandas_fundamentals.ipynb) — pandas fundamentals with a self-built dataset
- [`S8_Pandas_Advanced_and_Exercises.ipynb`](notebooks/S8_Pandas_Advanced_and_Exercises.ipynb) — Merge/join, string cleaning, dates, missing values, MultiIndex, and categorical dtype
- [`100_pandas_puzzles.ipynb`](notebooks/100_pandas_puzzles.ipynb) — Targeted practice from 100-pandas-puzzles

## Repository Structure

```
ml-learning/
├── notebooks/
│   ├── 100_Numpy_exercises.ipynb
│   ├── s2_svd_pca.ipynb
│   ├── S5_NumPy_Examples.ipynb
│   ├── S6_NumPy_Broadcasting_Axis_Linalg.ipynb
│   ├── s7_pandas_fundamentals.ipynb
│   ├── S8_Pandas_Advanced_and_Exercises.ipynb
│   └── 100_pandas_puzzles.ipynb
├── data/
├── pyproject.toml
├── uv.lock
└── README.md
```