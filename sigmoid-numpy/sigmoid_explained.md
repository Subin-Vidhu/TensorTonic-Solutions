# What the Sigmoid Function Does

The sigmoid function squashes any real number into the range $(0, 1)$:

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

No matter how large or small the input, the output is always between 0 and 1 (exclusive). This makes sigmoid ideal for representing probabilities.

---

## The Shape of Sigmoid

The sigmoid curve is an **S-shape** (the word "sigmoid" comes from the Greek letter sigma, which looks like an S):

- For large **negative** inputs: output approaches **0**
- For large **positive** inputs: output approaches **1**
- At $x = 0$: output is exactly **0.5**

Some concrete values:

| Input $x$ | $\sigma(x)$ |
|-----------|-------------|
| $-10$ | $\approx 0.000045$ |
| $-5$  | $\approx 0.0067$   |
| $-2$  | $\approx 0.119$    |
| $-1$  | $\approx 0.269$    |
| $0$   | $= 0.5$            |
| $1$   | $\approx 0.731$    |
| $2$   | $\approx 0.881$    |
| $5$   | $\approx 0.9933$   |
| $10$  | $\approx 0.999955$ |

---

## Why Sigmoid Outputs Probabilities

The formula $\frac{1}{1 + e^{-x}}$ guarantees:

- **Always positive:** $e^{-x} > 0$ for all $x$, so the denominator $1 + e^{-x} > 1$, making the fraction positive
- **Always less than 1:** The denominator is always greater than 1, so the fraction is less than 1
- **Monotonically increasing:** As $x$ increases, $e^{-x}$ decreases, so the fraction increases

These properties make sigmoid perfect for converting raw model outputs (logits) into probabilities for binary classification.

---

## The Derivative of Sigmoid

The derivative has an elegant form:

$$\frac{d\sigma}{dx} = \sigma(x) \cdot (1 - \sigma(x))$$

This means once you compute $\sigma(x)$, the gradient is essentially free to calculate.

Derivative values:

| $x$ | $\sigma'(x)$ | Note |
|-----|-------------|------|
| $0$ | $0.5 \times 0.5 = 0.25$ | Maximum |
| $2$ | $0.881 \times 0.119 \approx 0.105$ | |
| $5$ | $\approx 0.0066$ | Very small |

> The gradient is largest at $x = 0$ and quickly shrinks as you move away from zero. This causes the **vanishing gradient problem** in deep networks.

---

## Numerical Stability

Computing $e^{-x}$ directly can overflow for large negative $x$ (since $e^{-(-1000)} = e^{1000} = \infty$).

A numerically stable implementation handles positive and negative inputs differently:

**For $x \geq 0$:**

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

**For $x < 0$:**

$$\sigma(x) = \frac{e^x}{1 + e^x}$$

Both formulas are mathematically equivalent, but the second avoids computing $e^{-x}$ when $x$ is a large negative number.

---

## Sigmoid vs. Other Activations

| Activation | Range | Gradient at 0 | Common Use |
|------------|-------|---------------|------------|
| **Sigmoid** | $(0, 1)$ | $0.25$ | Output layers for binary classification, gating |
| **Tanh** | $(-1, 1)$ | $1.0$ | Hidden layers (legacy), LSTM/GRU internal states |
| **ReLU** | $[0, \infty)$ | $1.0$ for $x > 0$, $0$ for $x < 0$ | Hidden layers (modern default since 2012) |

**Tanh** is a rescaled sigmoid: $\tanh(x) = 2\sigma(2x) - 1$. It is zero-centered (outputs range from $-1$ to $1$), which often leads to faster convergence than sigmoid in hidden layers.

**ReLU** ($\max(0, x)$) solved the vanishing gradient problem for deep networks. Its gradient is either 0 or 1, so gradients flow without shrinking through the positive regime. It became the default hidden-layer activation starting around 2012.

Sigmoid remains the right choice for **outputs that need to represent probabilities** or for **gating mechanisms** that need smooth $[0, 1]$ control signals.

---

## Where Sigmoid Is Used Today

### Binary Classification Output Layer

The final layer of a binary classifier typically outputs a single logit, and sigmoid converts it to a probability:

$$P(y = 1 \mid x) = \sigma(\text{logit})$$

### Gating Mechanisms

In **LSTMs** and **GRUs**, sigmoid gates control information flow:

- **Forget gate:** decides what to discard from cell state
- **Input gate:** decides what new information to store
- **Output gate:** decides what to output

These gates need values in $[0, 1]$ to act as "soft switches" (0 = block, 1 = pass through).

### Attention Weights

Some attention mechanisms use sigmoid instead of softmax when attention weights should be **independent** (not sum to 1).

### Multi-label Classification

When each class is independent (an image can have multiple labels), apply sigmoid to each output **independently** rather than using softmax across all classes.
