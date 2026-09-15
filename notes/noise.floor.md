# Noise Floor Estimate

The target `p_recall` is a binomial proportion:

$$
y_i = \frac{K_i}{n_i},
\qquad
K_i \sim \mathrm{Binomial}(n_i, \pi_i)
$$

Therefore, even a perfect model that predicts the true recall probability $\pi_i$ has irreducible variance:

$$
\mathrm{Var}(y_i \mid \pi_i, n_i)
=
\frac{\pi_i(1-\pi_i)}{n_i}
$$

This gives the dataset-level noise-floor estimate:

$$
\mathrm{RMSE}_{\text{floor}}
=
\sqrt{
E\left[
\frac{\pi_i(1-\pi_i)}{n_i}
\right]
}
$$

## Constant-probability estimate

Using the global CV-pool mean

$$
\bar{\pi} = \bar{y} = 0.89424
$$

for every row gives:

- Estimated MSE floor: `0.07110`
- Estimated RMSE floor: `0.26664`

## OOF-based estimate

A 5-fold `GroupKFold` Ridge model with `user_id` grouping was used to generate out-of-fold predictions as proxies for $\pi_i$.

Predictions were clipped to $[0,1]$ because Ridge regression is not probability-constrained. Only 8 of 14,438 predictions exceeded 1 before clipping.

The resulting estimate was:

- Estimated MSE floor: `0.06997`
- Estimated RMSE floor: `0.26452`

The OOF-based estimate is slightly lower because it captures some heterogeneity in recall probabilities across rows instead of assigning the same probability to every observation.

## Interpretation

The tuned Ridge model achieves approximately `0.2735` CV RMSE, while the estimated noise floor is approximately `0.264–0.267`.

This leaves only about `0.007–0.009` RMSE between the current model and the estimated irreducible-error region.

This suggests that most of the remaining error is likely constrained by target noise rather than severe model underfitting, although some learnable signal may still remain.

The OOF-based estimate should not be interpreted as an exact theoretical lower bound. It is model-dependent because the true latent probabilities $\pi_i$ are unknown, so different models may produce different noise-floor estimates.