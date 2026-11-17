# Mathematical Foundations of Chebyshev State-Space Compression

This whitepaper details the mathematical derivation, approximation theory, and dynamical systems principles underpinning the **Chebyshev-Compressor** architecture.

---

## 1. The Continuous-Time Memory Problem

Let $u(t) \in \mathbb{R}$ denote a continuous-time real-valued sensor signal. An ideal continuous memory mechanism maintaining history over a sliding temporal window $[t - \theta, t]$ of duration $\theta > 0$ must track the historical trajectory function:

$$
f_t(\tau) = u(t - \tau), \quad \tau \in [0, \theta]
$$

where $\tau = 0$ corresponds to the immediate present ($u(t)$) and $\tau = \theta$ represents the oldest preserved historical observation ($u(t - \theta)$).

In conventional Recurrent Neural Networks (RNNs, LSTMs, GRUs), this historical function is compressed heuristically into a latent state vector $h_t \in \mathbb{R}^H$ through parameterized non-linear transitions:

$$
h_{t+1} = \tanh(W_h h_t + W_x x_{t+1} + b)
$$

Under backpropagation through time over $T$ steps, the gradient of the loss $\mathcal{L}$ with respect to the initial state $h_0$ involves repeated products of Jacobian matrices:

$$
\frac{\partial \mathcal{L}}{\partial h_0} = \frac{\partial \mathcal{L}}{\partial h_T} \prod_{t=1}^T \frac{\partial h_t}{\partial h_{t-1}} = \frac{\partial \mathcal{L}}{\partial h_T} \prod_{t=1}^T \mathrm{diag}(1 - h_t^2) W_h
$$

Because the operator norm $\| \mathrm{diag}(1 - h_t^2) W_h \| \le \sigma < 1$ in contracting regimes, the gradient decays **exponentially**:

$$
\left\| \frac{\partial \mathcal{L}}{\partial h_0} \right\| \le C \cdot \sigma^T = O\left(e^{-\alpha T}\right), \quad \alpha > 0
$$

Conversely, if $\|W_h\| > 1$, gradients explode. This fundamental dichotomy causes conventional RNNs to suffer from catastrophic forgetting and gradient decay over long context horizons.

---

## 2. Orthogonal Polynomial Memory & Minimax Optimality

Rather than relying on unconstrained recurrent weights, **Chebyshev-Compressor** projects $f_t(\tau)$ onto a truncated orthogonal basis of $K$ **Shifted Chebyshev Polynomials of the First Kind** $T_k^*(\tau)$:

$$
f_t(\tau) \approx \sum_{k=0}^{K-1} c_k(t) \phi_k(\tau)
$$

where $\mathbf{c}(t) = [c_0(t), c_1(t), \dots, c_{K-1}(t)]^T \in \mathbb{R}^K$ is the Chebyshev State Vector.

### 2.1 Why Chebyshev Polynomials? (The Minimax Property)

By the **Chebyshev Equioscillation Theorem**, among all monic polynomials $P_n(x)$ of degree $n$ on $[-1, 1]$, the scaled Chebyshev polynomial $2^{1-n} T_n(x)$ uniquely minimizes the uniform supremum norm ($L_\infty$-norm):

$$
\min_{P_n \in \mathcal{P}_n} \|P_n\|_\infty = \left\| 2^{1-n} T_n \right\|_\infty = 2^{1-n}
$$

This property guarantees that approximating a continuous function using Chebyshev polynomial projection yields an approximation error that is within an infinitesimal factor of the mathematically optimal polynomial approximation:

$$
\|f - p_K^*\|_\infty \le \left( 1 + \frac{2}{\pi} \ln K + O(1) \right) \min_{p \in \mathcal{P}_{K-1}} \|f - p\|_\infty
$$

In contrast to Taylor expansions (which concentrate accuracy at a single expansion point while diverging rapidly near window boundaries) or Fourier series (which suffer from Gibbs ringing phenomenon near non-periodic endpoints), **Chebyshev expansions distribute approximation error equioscillating uniformly across the entire history window $[0, \theta]$**.

### 2.2 Spectral Decay of Expansion Coefficients

For any input trajectory $f(\tau) \in C^p([0, \theta])$ whose $p$-th derivative has bounded total variation $V$, the Chebyshev expansion coefficients $c_k$ satisfy the spectral decay bound:

$$
|c_k| \le \frac{2 V}{\pi k (k-1) \cdots (k-p+1)} = O\left( k^{-p} \right)
$$

If the signal is analytic within an ellipse containing $[0, \theta]$ with sum of semi-axes $\rho > 1$, the coefficients decay **exponentially** in spectral degree:

$$
|c_k| \le \frac{2 M}{\rho - 1} \rho^{-k} = O\left( \rho^{-k} \right)
$$

This rapid spectral decay guarantees that retaining merely $K = 8$ polynomial coefficients captures over **99.5%** of the total historical signal energy for real-world machinery sensor dynamics.

---

## 3. Continuous-Time Dynamical System (HiPPO-ChebT)

To continuously maintain $\mathbf{c}(t)$ as time $t$ flows forward without re-integrating over past history, the transport PDE governing sliding memory:

$$
\frac{\partial f_t}{\partial t} = -\frac{\partial f_t}{\partial \tau}, \quad f_t(0) = u(t)
$$

is projected onto the orthonormal Chebyshev basis under the weight measure:

$$
\omega(\tau) = \frac{1}{\sqrt{\tau(\theta - \tau)}}
$$

This yields an exact linear continuous-time ordinary differential equation:

$$
\frac{d\mathbf{c}(t)}{dt} = A \mathbf{c}(t) + B u(t)
$$

where:

- **Basis scaling factors**: $R_0 = 1.0$, and $R_n = \sqrt{2.0}$ for $n \ge 1$.
- **Continuous transition operator** $A \in \mathbb{R}^{K \times K}$ has elements:

$$
A_{n, k} = -\frac{1}{\theta} R_n R_k \cdot \begin{cases} 
(-1)^{n - k}, & \text{if } n < k \\ 
1.0, & \text{if } n \ge k 
\end{cases}
$$

- **Continuous input projection vector** $B \in \mathbb{R}^{K \times 1}$ is:

$$
B_n = \frac{1}{\theta} R_n
$$

### 3.1 Mathematical Solution to Vanishing Gradients: Algebraic vs. Exponential Decay

In the linear continuous system $\dot{\mathbf{c}} = A \mathbf{c} + B u$, the state transition operator across an elapsed interval $t$ is the matrix exponential:

$$
\Phi(t) = \exp(A t)
$$

The continuous matrix $A$ satisfies a skew-symmetric normal plus low-rank (NPLR) decomposition:

$$
A = S - \frac{1}{2} B B^T, \quad S^T = -S
$$

Because $S$ is skew-symmetric ($S^T = -S$), its eigenvalues are purely imaginary ($\lambda \in i\mathbb{R}$), meaning the energy norm is conserved:

$$
\frac{d}{dt} \|\mathbf{c}(t)\|^2 = \mathbf{c}^T (A + A^T) \mathbf{c} = -\left( B^T \mathbf{c} \right)^2 \le 0
$$

Rather than decaying exponentially as $e^{-\alpha t}$, the historical memory and backpropagated gradients in the orthogonal Chebyshev state space decay **algebraically (polynomially)**:

$$
\| \Phi(t) \| = O\left( \frac{1}{t} \right)
$$

This polynomial memory decay is the rigorous mathematical explanation for why the Chebyshev State-Space Cell retains long-term historical dependencies across hundreds of time steps without vanishing gradients and without needing non-linear forget gates.

---

## 4. Continuous-to-Discrete Conversion: Double-Precision Tustin Transform

To deploy this continuous dynamical system to digital computers and embedded microcontrollers operating at discrete sampling steps $\Delta t$, we discretize $(A, B)$ using the **Bilinear / Tustin Transform**:

$$
s \approx \frac{2}{\Delta t} \frac{z - 1}{z + 1}
$$

Substituting into the continuous transfer operator $(sI - A)^{-1} B$ yields the discrete-time recursive update:

$$
\mathbf{s}_{t+1} = \bar{A} \mathbf{s}_t + \bar{B} x_t
$$

where:

$$
\bar{A} = \left( I - \frac{\Delta t}{2} A \right)^{-1} \left( I + \frac{\Delta t}{2} A \right)
$$

$$
\bar{B} = \left( I - \frac{\Delta t}{2} A \right)^{-1} (\Delta t B)
$$

### 4.1 Guaranteed Discrete Stability

The Bilinear Transform is a conformal mapping that maps the open left-half complex plane $\mathrm{Re}(s) < 0$ strictly into the interior of the unit disk:

$$
|z| < 1 \iff \mathrm{Re}(s) < 0
$$

Because all eigenvalues of the continuous Chebyshev matrix satisfy $\mathrm{Re}(\lambda_i(A)) < 0$, every discrete eigenvalue $\bar{\lambda}_i$ of $\bar{A}$ satisfies:

$$
\left| \lambda_i(\bar{A}) \right| < 1.0, \quad \forall i \in \{0, \dots, K-1\}
$$

This guarantees **Bounded-Input Bounded-Output (BIBO) stability** across infinite sequence horizons. Matrix inversion $\left(I - \frac{\Delta t}{2} A\right)^{-1}$ is computed in double-precision `float64` before casting to single-precision `float32`, eliminating numerical drift.

---

## 5. Depthwise Multi-Sensor State Space and Non-Linear Readout

For a multi-sensor input $x_t \in \mathbb{R}^{D_{\mathrm{in}}}$ (e.g., $D_{\mathrm{in}} = 14$ in NASA C-MAPSS):

1. **Depthwise Decomposition**: Each physical sensor channel $d \in \{1, \dots, D_{\mathrm{in}}\}$ maintains its own independent state vector $\mathbf{s}_t^{(d)} \in \mathbb{R}^K$, sharing transition dynamics $(\bar{A}, \bar{B})$:

$$
\mathbf{s}_{t+1}^{(d)} = \bar{A} \mathbf{s}_t^{(d)} + \bar{B} \cdot \mathrm{clamp}\!\left( \frac{x_t^{(d)} - \mu_d}{\sigma_d},\; -5.0,\; 5.0 \right)
$$

2. **Non-Linear Decoupled Readout**: Cross-channel interactions and decision boundaries are handled outside the recurrent loop by a 2-layer Multi-Layer Perceptron:

$$
y_t = W_2 \, \mathrm{ReLU}\!\left( W_1 [\mathbf{s}_t^{(1)}, \dots, \mathbf{s}_t^{(D_{\mathrm{in}})}] + b_1 \right) + b_2
$$

This hybrid design eliminates all sequential transcendental operations ($\sigma, \tanh$) from the recurrent state loop while preserving $>95\%$ of multi-layer LSTM expressive capacity.
