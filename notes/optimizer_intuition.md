# Optimizer Intuition

## Convex vs Non-Convex

A convex loss surface does not contain bad local minima: every local minimum is also a global minimum. Linear regression with squared error is convex, which makes gradient-based optimization easier to reason about.

Non-convex objectives can contain multiple local minima, saddle points, and flat regions, so gradient descent may converge to different solutions depending on initialization and optimization dynamics.

## Batch GD, SGD, and Mini-Batch GD

Batch Gradient Descent computes each update using the entire training set. This produces stable, low-noise gradients, but each parameter update becomes expensive when the dataset is large.

Stochastic Gradient Descent uses a single training example for each update. Its updates are cheap and frequent, but the gradient is noisy, so the optimization path tends to fluctuate around the direction of the full gradient.

Mini-batch Gradient Descent uses a small subset of the training set for each update. It provides a compromise between the stability of Batch GD and the efficiency of SGD, and is especially useful for large datasets and neural network training.

## Momentum

Momentum keeps a moving average of previous gradients:

\[
v_t = \beta v_{t-1} + g_t
\]

\[
w_t = w_{t-1} - \alpha v_t
\]

The main idea is to accumulate movement in directions where gradients are consistently aligned, while reducing oscillation in directions where the gradient repeatedly changes sign.

This is especially useful in narrow valleys, where ordinary gradient descent may zig-zag across a steep direction while making slow progress along a flatter direction.

## Adam

Adam combines two ideas:

1. A first-moment estimate, similar to momentum, which tracks the average gradient direction.
2. A second-moment estimate, which tracks the average squared gradient magnitude.

\[
m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t
\]

\[
v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2
\]

After bias correction, the update is approximately:

\[
w_t =
w_{t-1}
-
\alpha
\frac{\hat{m}_t}
{\sqrt{\hat{v}_t}+\epsilon}
\]

This gives Adam adaptive step sizes for individual parameters. Parameters with consistently large gradients receive relatively smaller effective steps, while parameters with smaller gradients are not suppressed as strongly.

Adam can therefore combine directional consistency from momentum with adaptive scaling of the learning rate.

## Narrow Valley Experiment

In the toy loss

\[
J(w_1,w_2)
=
\frac{1}{2}
\left(
0.1w_1^2 + 10w_2^2
\right)
\]

the curvature in the \(w_2\) direction is much larger than in the \(w_1\) direction.

Batch Gradient Descent produced strong oscillation in the steep \(w_2\) direction while progressing slowly along \(w_1\).

Momentum accumulated consistent movement along the flat direction and reached the minimum much faster, although it could overshoot because of the accumulated velocity.

Adam produced more controlled updates in the high-gradient direction because its second-moment estimate reduced the effective step size for parameters with large gradients.

The experiment showed that optimizer behavior depends strongly on the geometry of the loss surface.

## Flagship Decision

Batch gradient descent was reasonable because our training dataset is relatively small, with only 14,438 samples and 5 features, so computing the full gradient at each step is inexpensive.

Mini-batch gradient descent would become a better choice if the dataset or model grew large enough that computing the full gradient for every update became costly.

Optimizer choice is an engineering decision. Adam, Momentum, SGD, and Batch GD each have different trade-offs, and Adam is not automatically the best choice for every problem.