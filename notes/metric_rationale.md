# Metric Rationale

## Evaluation Metric

The primary regression metric is RMSE.

RMSE penalizes larger prediction errors more strongly because errors are squared before averaging.

## Group-Aware Evaluation

The dataset contains multiple observations from the same `user_id`.

A random row-level split could place the same user in both training and validation sets, causing group leakage.

Therefore, evaluation uses 5-fold `GroupKFold` with `user_id` as the grouping variable.

This ensures:

- train and validation users are disjoint
- validation users are unseen during training
- preprocessing is fit only on the training fold through a `Pipeline`

## Month 2 Results

5-fold group-aware evaluation:

- Mean baseline RMSE: `0.275398 ± 0.010158`
- Linear regression RMSE: `0.273699 ± 0.009834`

The linear model slightly outperforms the mean baseline, indicating that the current features contain some generalizable predictive signal.

The improvement is small, so this is evidence of an initial signal rather than a strong model.

`mean ± std` summarizes average performance and fold-to-fold variability. It is not a confidence interval.