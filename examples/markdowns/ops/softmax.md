# Softmax Activation

Softmax turns a vector of $K$ real values into a vector of $K$ real values that sum to 1.0. Each output can be interpreted as a probability.

## Formula
For a given input vector $z$, the Softmax function is defined as:

$$\sigma(z)_i = \frac{e^{z_i}}{\sum_{j=1}^K e^{z_j}}$$

## Numerical Stability Trick
Computing $e^{z_i}$ directly can lead to overflow (e.g., $e^{1000}$ is too large for a float). To prevent this, we subtract the maximum value from the input vector before exponentiation:

$$\sigma(z)_i = \frac{e^{z_i - \max(z)}}{\sum e^{z_j - \max(z)}}$$

This shift does not change the result but ensures that all values passed to `exp()` are $\le 0$, keeping the output between $0$ and $1$.