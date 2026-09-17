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

## Regularized Models

Ridge and Lasso were tuned with `GridSearchCV` under the same protocol: 5-fold `GroupKFold`, `groups=user_id` passed explicitly, and `StandardScaler` inside the pipeline so scaling statistics come only from each training fold.

`GridSearchCV` maximizes the score, so `neg_root_mean_squared_error` was used and the sign flipped for reporting.

Ridge searched `[0.01, 0.1, 1, 3, 10, 30, 100, 300]` and selected `alpha = 30`.
Lasso searched values from `0.00001` to `0.1` and selected `alpha = 0.0001`.

| Model | CV RMSE |
|---|---|
| Mean baseline | 0.275398 ± 0.010158 |
| LinearRegression | 0.273699 ± 0.009834 |
| Ridge, `alpha = 30` | 0.273468 ± 0.010290 |
| Lasso, `alpha = 0.0001` | 0.273533 ± 0.010116 |

Ridge has the best mean CV score. The gap to ordinary linear regression is 0.00023 in the mean, against a fold-to-fold standard deviation of about 0.010.

Those two numbers should not be set against each other. Every model was scored on the same folds, so the informative comparison is paired: take the per-fold difference and summarise it as `mean(delta) ± std(delta)`.

That comparison is available for the mean baseline against linear regression, and it changes the reading. Linear regression wins on all five folds, with `mean(delta) = -0.001699 ± 0.000576`. The improvement is small but consistent, not noise.

It is not available for Ridge or Lasso, because `GridSearchCV` returned only summary statistics and the per-fold scores were not kept. Their margins are therefore uncalled rather than disproved. Keeping per-fold scores is a process fix for the next month.

Neither of these is a significance test, and neither should be reported as one.

Carrying Ridge forward is therefore not a performance claim. The reason to prefer a penalised fit at all is the 0.999 collinearity between `history_seen` and `history_correct`, which leaves unpenalised coefficients unstable even where predictions are not.

The selected Lasso alpha is the smallest value in the grid, which means the search preferred almost no sparsity. Stronger sparsity made the score worse. At `alpha = 0.1` all five coefficients are driven to zero, the model reduces to its intercept, and the CV score becomes `0.275398` — exactly the mean baseline. That is a useful check: a fully sparse linear model and a mean baseline are the same model.

Sparsity is not helpful for a five-feature representation in which every feature carries some signal.

## Hyperparameter Selection

Alpha was selected on the CV pool only. The holdout was not touched at any point during tuning.

Using the holdout score to choose a hyperparameter would turn the holdout into a tuning signal, and the final evaluation would no longer be independent.

Parameters (`w`, `b`) are learned from the training data. Hyperparameters (`alpha`) are chosen outside training, using cross-validation.

## Month 3 — Evaluation Diagnostics

Andrew's train/CV/test framework maps directly to the flagship project, but the user-level structure requires group-aware evaluation. Random row-level splitting could place the same user in both training and validation data, giving the model information about users it is later asked to predict. Therefore, validation uses `GroupKFold` with `user_id`.

Bias and variance should be diagnosed relative to different references:

- High bias: training error is high relative to the baseline level of performance.
- High variance: validation error is substantially worse than training error.
- More data generally helps high variance more than high bias.
- Stronger models, better features, or less regularization can help high bias.
- More regularization, simpler models, or more data can help high variance.

The tuned Ridge model reaches roughly `0.2735` CV RMSE, while the estimated noise floor is around `0.264–0.267`. This suggests that the remaining achievable improvement may be relatively small, so model performance should not be judged from RMSE alone without considering the problem's irreducible error.

Learning curves provide another diagnostic by showing how training and validation errors change as the amount of training data increases.

Finally, reporting `mean ± fold std` is preferable to reporting a single CV split because it shows both average performance and sensitivity to the chosen folds. Fold standard deviation is not a confidence interval and should not be interpreted as uncertainty of the CV mean.

## Month 3 — ML Development Process

The ML development process should be iterative: choose a direction, train the model, diagnose its errors, and use those diagnostics to decide the next experiment.

Error analysis helps prioritize work by identifying which error categories contribute most to overall performance loss.

More data is not automatically useful. It is most valuable when diagnostics suggest variance or when additional data targets specific failure modes.

Data augmentation can create useful training variation while preserving labels.

Precision, recall, and F1 are important for imbalanced classification problems, but they are not appropriate primary metrics for the current `p_recall` regression task.

The flagship project therefore continues to use group-aware CV and RMSE as its main evaluation framework.

The next step is regression-oriented error analysis: inspect OOF residuals and determine where the Ridge model makes its largest errors.