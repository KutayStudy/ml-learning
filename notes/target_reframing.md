# Target Reframing

The current target is

$$
p\_recall = \frac{session\_correct}{session\_seen}
$$

and is trained with ordinary MSE.

This ignores that target reliability depends strongly on `session_seen`. In particular,

$$
Var(p\_recall \mid \pi,n)
=
\frac{\pi(1-\pi)}{n}
$$

so small-denominator observations are much noisier.

Weighted regression could partially address this by giving larger `session_seen` rows more weight, but the weighting rule would still be manually chosen.

A more natural formulation for the current aggregated dataset is:

$$
session\_correct
\sim
Binomial(session\_seen,\pi)
$$

and training with Binomial negative log-likelihood.

If genuine trial-level data were available, the natural formulation would instead be Bernoulli modeling with binary cross-entropy.

For now, `p_recall + RMSE` remains the baseline, while Binomial-aware modeling is the preferred alternative formulation.