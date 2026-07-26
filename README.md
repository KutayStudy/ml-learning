# ML Learning

A lightweight, notebook-first repository for learning the mathematical and practical foundations of machine learning.

The repository focuses on runnable experiments, small exercises, and practical learning artifacts rather than extensive standalone notes.

## Mathematical Foundations

- **SVD and PCA:** Singular vectors, singular values, low-rank approximation, explained variance, and dimensionality reduction.
- **Maximum Likelihood Estimation:** Choosing parameters that maximize the likelihood of the observed data.
- **MAP and Regularization:** Combining likelihood with prior knowledge; Gaussian priors lead to L2 regularization and Ridge regression.

## Numerical Computing

- **NumPy Fundamentals:** Array creation, data types, indexing, slicing, Boolean masking, fancy indexing, reshaping, flattening, transposition, and vectorized operations.
- **NumPy Practice:** Completed exercises 1–30 from the [numpy-100](https://github.com/rougier/numpy-100) collection.

## Notebooks

- [`s2_svd_pca.ipynb`](notebooks/s2_svd_pca.ipynb) — SVD, low-rank approximation, and PCA
- [`S5_NumPy_Examples.ipynb`](notebooks/S5_NumPy_Examples.ipynb) — NumPy fundamentals with runnable examples
- [`100_Numpy_exercises.ipynb`](notebooks/100_Numpy_exercises.ipynb) — Solutions to numpy-100 exercises 1–30

## Repository Structure

```text
ml-learning/
├── notebooks/
│   ├── 100_Numpy_exercises.ipynb
│   ├── s2_svd_pca.ipynb
│   └── S5_NumPy_Examples.ipynb
├── data/
├── pyproject.toml
├── uv.lock
└── README.md