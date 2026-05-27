
# Definition

[[Linear State Space Model]]
$$
\begin{cases}
x_t = \phi x_{t-1} + b + \epsilon_t \\
y_t = \alpha x_t + d + \eta_t \\
\epsilon_t \sim \mathcal{N}(0, \sigma^2) \\
\eta_t \sim \mathcal{N}(0, \tau^2)
\end{cases}
$$
In other words,

$$
x_i \mid x_{i-1}
\sim
\mathcal{N}( \phi x_{i-1}+b, \sigma^2),
$$
and
$$
y_i \mid x_i
\sim
\mathcal{N}( \alpha x_i+d, \tau^2).
$$

### Stationary distribution

Assuming the initial distribution is the stationary distribution of the latent process, we have

$$
x_0 \sim \mathcal{N}(\mu_0,P_0),
$$

where
$$
\mu_0 = \frac{b}{1-\phi}
$$
and
$$
P_0 = \frac{\sigma^2}{1-\phi^2}.
$$
This requires
$$
|\phi|<1.
$$

### Joint Density

$$
p(x_{0:t},y_{0:t}\mid \theta)
=
p(x_0\mid \theta)
\prod_{i=1}^t p(x_i\mid x_{i-1},\theta)
\prod_{i=0}^t p(y_i\mid x_i,\theta).
$$

### Log-Likelihood

The complete-data log-likelihood is

$$
\begin{align}
\log p(x_{0:t},y_{0:t}\mid\theta)
&=
-\frac{1}{2}\log(2\pi P_0)
-\frac{(x_0-\mu_0)^2}{2P_0}
\\
&\quad
-
\sum_{i=1}^t
\left[
\frac{1}{2}\log(2\pi\sigma^2)
+
\frac{(x_i-\phi x_{i-1}-b)^2}{2\sigma^2}
\right]
\\
&\quad
-
\sum_{i=0}^t
\left[
\frac{1}{2}\log(2\pi\tau^2)
+
\frac{(y_i-\alpha x_i-d)^2}{2\tau^2}
\right].
\end{align}
$$

### Residuals

Define the transition residuals

$$
e_i = x_i-\phi x_{i-1}-b
$$

and the observation residuals

$$
u_i = y_i-\alpha x_i-d.
$$

Using the residuals, the complete-data log-likelihood can be written compactly as

$$
\begin{align}
\log p(x_{0:t},y_{0:t}\mid\theta)
&=
-\frac{1}{2}\log(2\pi P_0)
-\frac{(x_0-\mu_0)^2}{2P_0}
\\
&\quad
-
\sum_{i=1}^t
\left[
\frac{1}{2}\log(2\pi\sigma^2)
+
\frac{e_i^2}{2\sigma^2}
\right]
\\
&\quad
-
\sum_{i=0}^t
\left[
\frac{1}{2}\log(2\pi\tau^2)
+
\frac{u_i^2}{2\tau^2}
\right].
\end{align}
$$

### Parameter Scores for Joint Log-Likelihood

Ignoring the derivative of the stationary initial density for simplicity, the complete-data score terms are
$$
\frac{\partial}{\partial \phi}
\log p(x_{0:t},y_{0:t}\mid\theta)
=
\sum_{i=1}^t
\frac{e_i x_{i-1}}{\sigma^2},
$$

$$
\frac{\partial}{\partial b}
\log p(x_{0:t},y_{0:t}\mid\theta)
=
\sum_{i=1}^t
\frac{e_i}{\sigma^2},
$$

$$
\frac{\partial}{\partial \sigma^2}
\log p(x_{0:t},y_{0:t}\mid\theta)
=
-\frac{t}{2\sigma^2}
+
\frac{1}{2(\sigma^2)^2}
\sum_{i=1}^t e_i^2,
$$

$$
\frac{\partial}{\partial \alpha}
\log p(x_{0:t},y_{0:t}\mid\theta)
=
\sum_{i=0}^t
\frac{u_i x_i}{\tau^2},
$$

$$
\frac{\partial}{\partial d}
\log p(x_{0:t},y_{0:t}\mid\theta)
=
\sum_{i=0}^t
\frac{u_i}{\tau^2},
$$

$$
\frac{\partial}{\partial \tau^2}
\log p(x_{0:t},y_{0:t}\mid\theta)
=
-\frac{t+1}{2\tau^2}
+
\frac{1}{2(\tau^2)^2}
\sum_{i=0}^t u_i^2.
$$


If we include the derivative of the stationary initial density, then

$$
\mu_0 = \frac{b}{1-\phi},
\qquad
P_0 = \frac{\sigma^2}{1-\phi^2}.
$$

For any scalar parameter $\theta_j$,

$$
\frac{\partial}{\partial \theta_j}
\log p(x_0\mid\theta)
=
-\frac{1}{2P_0}
\frac{\partial P_0}{\partial \theta_j}
+
\frac{x_0-\mu_0}{P_0}
\frac{\partial \mu_0}{\partial \theta_j}
+
\frac{(x_0-\mu_0)^2}{2P_0^2}
\frac{\partial P_0}{\partial \theta_j}.
$$

The needed derivatives are

$$
\frac{\partial \mu_0}{\partial \phi}
=
\frac{b}{(1-\phi)^2},
\qquad
\frac{\partial \mu_0}{\partial b}
=
\frac{1}{1-\phi},
\qquad
\frac{\partial \mu_0}{\partial \sigma^2}
=
0,
$$

and

$$
\frac{\partial P_0}{\partial \phi}
=
\frac{2\phi\sigma^2}{(1-\phi^2)^2},
\qquad
\frac{\partial P_0}{\partial \sigma^2}
=
\frac{1}{1-\phi^2},
\qquad
\frac{\partial P_0}{\partial b}
=
0.
$$



### Observation Data Likelihood (Integrating out Latent State)


The observed-data likelihood is obtained by integrating out the latent path:
$$
p(y_{0:t}\mid \theta)
=
\int p(x_{0:t},y_{0:t}\mid \theta)\,dx_{0:t}.
$$
For the linear-Gaussian state-space model, the marginal likelihood can be computed exactly using the Kalman filter.

Assume the parameter vector $\theta = (\phi,b,\sigma^2,\alpha,d,\tau^2)$ satisfies the constraints
$$
|\phi|<1,
\qquad
\sigma^2>0,
\qquad
\tau^2>0.
$$
And assume the stationary initialization
$$
x_0 \sim \mathcal{N}(\mu_0,P_0),
$$
as before.

---

### Kalman Prediction Step

Suppose that after observing $y_{0:i-1}$,

$$
x_{i-1}\mid y_{0:i-1}
\sim
\mathcal{N}(m_{i-1|i-1},P_{i-1|i-1}).
$$

The one-step-ahead predictive distribution of the state is
$$
x_i\mid y_{0:i-1}
\sim
\mathcal{N}(m_{i|i-1},P_{i|i-1}),
$$
where
$$
m_{i|i-1}
=
\phi m_{i-1|i-1}+b,
$$

and
$$
P_{i|i-1}
=
\phi^2 P_{i-1|i-1}+\sigma^2.
$$
---

### Predictive Distribution of the Observation

Because
$$
y_i = \alpha x_i+d+\eta_i,
$$
we have
$$
y_i\mid y_{0:i-1}
\sim
\mathcal{N}(\widehat{y}_{i|i-1},S_i),
$$
where
$$
\widehat{y}_{i|i-1}
=
\alpha m_{i|i-1}+d,
$$
and
$$
S_i
=
\alpha^2 P_{i|i-1}+\tau^2.
$$

Define the innovation
$$
v_i
=
y_i-\widehat{y}_{i|i-1}.
$$
Then
$$
v_i\mid y_{0:i-1}
\sim
\mathcal{N}(0,S_i).
$$

---

### Kalman Update Step

The Kalman gain is
$$
K_i
=
\frac{P_{i|i-1}\alpha}{S_i}.
$$
The filtering mean is
$$
m_{i|i}
=
m_{i|i-1}+K_i v_i.
$$

The filtering variance is

$$
P_{i|i}
=
(1-K_i\alpha)P_{i|i-1}.
$$

Equivalently, the Joseph-form variance update is

$$
P_{i|i}
=
(1-K_i\alpha)^2P_{i|i-1}
+
K_i^2\tau^2.
$$

The Joseph form is numerically more stable and preserves nonnegativity better.

---

### Marginal Likelihood

Using the chain rule,
$$
p(y_{0:t}\mid \theta)
=
\prod_{i=0}^t
p(y_i\mid y_{0:i-1},\theta),
$$
where $y_{0:-1}$ means no previous observations.

Each predictive density is Gaussian:
$$
p(y_i\mid y_{0:i-1},\theta)
=
\varphi(y_i\mid \widehat{y}_{i|i-1},S_i).
$$
Therefore,
$$
p(y_{0:t}\mid \theta)
=
\prod_{i=0}^t
\frac{1}{\sqrt{2\pi S_i}}
\exp
\left(
-\frac{v_i^2}{2S_i}
\right).
$$
The log-likelihood is
$$
\ell(\theta;y_{0:t})
=
\log p(y_{0:t}\mid \theta)
=
-\frac{1}{2}
\sum_{i=0}^t
\left[
\log(2\pi)
+
\log S_i
+
\frac{v_i^2}{S_i}
\right].
$$
This is called the **Kalman filter innovation likelihood**.

---

### MLE After Integrating Out the Latent State

The observed-data MLE is
$$
\widehat{\theta}_{\mathrm{MLE}}
=
\arg\max_{\theta}
\ell(\theta;y_{0:t}).
$$

Equivalently,
$$
\widehat{\theta}_{\mathrm{MLE}}
=
\arg\min_{\theta}
\frac{1}{2}
\sum_{i=0}^t
\left[
\log S_i
+
\frac{v_i^2}{S_i}
\right],
$$
ignoring the constant term
$$
\frac{t+1}{2}\log(2\pi).
$$

Unlike the complete-data MLE, the observed-data MLE usually does not have a simple closed-form solution because the innovations $v_i$ and innovation variances $S_i$ depend recursively on $\theta$ through the Kalman filter.

Therefore, the MLE is typically computed by numerical optimization:

$$
\widehat{\theta}_{\mathrm{MLE}}
=
\arg\min_{\theta}
\left\{
\frac{1}{2}
\sum_{i=0}^t
\left[
\log S_i
+
\frac{v_i^2}{S_i}
\right]
\right\}.
$$

Common constrained parametrizations are

$$
\phi = \tanh(\rho),
$$
$$
\sigma^2 = \exp(s),
$$
$$
\tau^2 = \exp(r),
$$

so that optimization can be performed over unconstrained parameters

$$
(\rho,b,s,\alpha,d,r)\in \mathbb{R}^6.
$$





# Observed Likelihood as a Marginal Gaussian Density

Consider the zero-mean 1D linear-Gaussian state-space model

$$
\begin{cases}
x_t = \phi x_{t-1} + \epsilon_t, \\
y_t = \alpha x_t + \eta_t, \\
\epsilon_t \sim \mathcal{N}(0,\sigma^2), \\
\eta_t \sim \mathcal{N}(0,\tau^2).
\end{cases}
$$

Assume

$$
|\phi|<1.
$$

Under the stationary distribution,

$$
x_t \sim \mathcal{N}\left(0,\frac{\sigma^2}{1-\phi^2}\right).
$$

For $s,t \in \{0,\dots,T\}$,

$$
\operatorname{Cov}(x_s,x_t)
=
\phi^{|t-s|}
\frac{\sigma^2}{1-\phi^2}.
$$

Since

$$
y_t = \alpha x_t+\eta_t,
$$

we have

$$
\mathbb{E}[y_t]=0
$$

and

$$
\operatorname{Cov}(y_s,y_t)
=
\alpha^2
\phi^{|t-s|}
\frac{\sigma^2}{1-\phi^2}
+
\tau^2\mathbf{1}_{s=t}.
$$

Define the observed vector

$$
y =
(y_0,\dots,y_T)^\top.
$$

Then

$$
y \sim \mathcal{N}(0,\Sigma_\theta),
$$

where

$$
(\Sigma_\theta)_{st}
=
\alpha^2
\phi^{|t-s|}
\frac{\sigma^2}{1-\phi^2}
+
\tau^2\mathbf{1}_{s=t}.
$$

Therefore, the observed-data likelihood is

$$
p(y_{0:T}\mid \theta)
=
\frac{1}{(2\pi)^{(T+1)/2}|\Sigma_\theta|^{1/2}}
\exp
\left(
-\frac{1}{2}y^\top \Sigma_\theta^{-1}y
\right).
$$

The observed-data log-likelihood is

$$
\ell(\theta;y)
=
\log p(y_{0:T}\mid \theta)
=
-\frac{T+1}{2}\log(2\pi)
-\frac{1}{2}\log|\Sigma_\theta|
-\frac{1}{2}y^\top \Sigma_\theta^{-1}y.
$$

# Score of the Observed Likelihood

For any scalar parameter $\theta_j$, the score is

$$
\frac{\partial \ell(\theta;y)}{\partial \theta_j}
=
-\frac{1}{2}
\operatorname{tr}
\left(
\Sigma_\theta^{-1}
\frac{\partial \Sigma_\theta}{\partial \theta_j}
\right)
+
\frac{1}{2}
y^\top
\Sigma_\theta^{-1}
\frac{\partial \Sigma_\theta}{\partial \theta_j}
\Sigma_\theta^{-1}
y.
$$

Equivalently,

$$
\boxed{
\frac{\partial \ell(\theta;y)}{\partial \theta_j}
=
\frac{1}{2}
\left[
y^\top
\Sigma_\theta^{-1}
\frac{\partial \Sigma_\theta}{\partial \theta_j}
\Sigma_\theta^{-1}
y
-
\operatorname{tr}
\left(
\Sigma_\theta^{-1}
\frac{\partial \Sigma_\theta}{\partial \theta_j}
\right)
\right].
}
$$

This is the score after integrating out the latent state.

# Derivatives of the Marginal Covariance

Let

$$
\gamma_k(\theta)
=
\phi^{|k|}
\frac{\sigma^2}{1-\phi^2},
\qquad
k=t-s.
$$

Then

$$
(\Sigma_\theta)_{st}
=
\alpha^2 \gamma_{t-s}(\theta)
+
\tau^2\mathbf{1}_{s=t}.
$$

The derivatives are

$$
\frac{\partial (\Sigma_\theta)_{st}}{\partial \alpha}
=
2\alpha
\phi^{|t-s|}
\frac{\sigma^2}{1-\phi^2},
$$

$$
\frac{\partial (\Sigma_\theta)_{st}}{\partial \sigma^2}
=
\alpha^2
\frac{\phi^{|t-s|}}{1-\phi^2},
$$

$$
\frac{\partial (\Sigma_\theta)_{st}}{\partial \tau^2}
=
\mathbf{1}_{s=t}.
$$

For $\phi$, write

$$
k = |t-s|.
$$

Then

$$
\frac{\partial}{\partial \phi}
\left(
\frac{\phi^k}{1-\phi^2}
\right)
=
\frac{
k\phi^{k-1}(1-\phi^2)
+
2\phi^{k+1}
}{
(1-\phi^2)^2
}.
$$

Therefore,

$$
\frac{\partial (\Sigma_\theta)_{st}}{\partial \phi}
=
\alpha^2\sigma^2
\frac{
k\phi^{k-1}(1-\phi^2)
+
2\phi^{k+1}
}{
(1-\phi^2)^2
},
\qquad
k=|t-s|.
$$

For $k=0$, this reduces to

$$
\frac{\partial (\Sigma_\theta)_{ss}}{\partial \phi}
=
\alpha^2\sigma^2
\frac{2\phi}{(1-\phi^2)^2}.
$$
# Closed-Form Score for the Observed Gaussian Likelihood

Let

$$
y=(y_0,\dots,y_T)^\top,
\qquad
\Sigma=\Sigma_\theta,
\qquad
M=\Sigma^{-1},
$$

and define

$$
z = \Sigma^{-1}y.
$$

The observed-data log-likelihood is

$$
\ell(\theta;y)
=
-\frac{T+1}{2}\log(2\pi)
-\frac{1}{2}\log|\Sigma_\theta|
-\frac{1}{2}y^\top\Sigma_\theta^{-1}y.
$$

For any scalar parameter $\theta_j$, the score is

$$
\frac{\partial \ell}{\partial \theta_j}
=
\frac{1}{2}
\left[
z^\top
\frac{\partial \Sigma}{\partial \theta_j}
z
-
\operatorname{tr}
\left(
M\frac{\partial \Sigma}{\partial \theta_j}
\right)
\right].
$$

Equivalently,

$$
\frac{\partial \ell}{\partial \theta_j}
=
\frac{1}{2}
\operatorname{tr}
\left[
\left(
zz^\top - M
\right)
\frac{\partial \Sigma}{\partial \theta_j}
\right].
$$

Therefore,

$$
\nabla_\theta \ell(\theta;y)
=
\frac{1}{2}
\begin{pmatrix}
\operatorname{tr}\left[\left(zz^\top-M\right)\partial_\phi\Sigma\right]
\\[0.8em]
\operatorname{tr}\left[\left(zz^\top-M\right)\partial_\alpha\Sigma\right]
\\[0.8em]
\operatorname{tr}\left[\left(zz^\top-M\right)\partial_{\sigma^2}\Sigma\right]
\\[0.8em]
\operatorname{tr}\left[\left(zz^\top-M\right)\partial_{\tau^2}\Sigma\right]
\end{pmatrix}.
$$
# Score Components

For $s,t\in\{0,\dots,T\}$, define

$$
k_{st}=|t-s|.
$$

The covariance matrix is

$$
(\Sigma_\theta)_{st}
=
\alpha^2
\frac{\sigma^2}{1-\phi^2}
\phi^{k_{st}}
+
\tau^2\mathbf{1}_{s=t}.
$$

Let

$$
M=\Sigma_\theta^{-1},
\qquad
z=M y,
\qquad
B=zz^\top-M.
$$

Then the score can be written as

$$
\frac{\partial \ell}{\partial \theta_j}
=
\frac{1}{2}
\operatorname{tr}
\left(
B\frac{\partial \Sigma_\theta}{\partial \theta_j}
\right).
$$

Equivalently, in elementwise form,

$$
\frac{\partial \ell}{\partial \theta_j}
=
\frac{1}{2}
\sum_{s=0}^T
\sum_{t=0}^T
B_{st}
\frac{\partial (\Sigma_\theta)_{st}}{\partial \theta_j}.
$$

The derivative with respect to $\alpha$ is

$$
\frac{\partial (\Sigma_\theta)_{st}}{\partial \alpha}
=
2\alpha
\frac{\sigma^2}{1-\phi^2}
\phi^{k_{st}}.
$$

Therefore,

$$
\boxed{
\frac{\partial \ell}{\partial \alpha}
=
\alpha
\frac{\sigma^2}{1-\phi^2}
\sum_{s=0}^T
\sum_{t=0}^T
B_{st}\phi^{k_{st}}.
}
$$

The derivative with respect to $\sigma^2$ is

$$
\frac{\partial (\Sigma_\theta)_{st}}{\partial \sigma^2}
=
\alpha^2
\frac{\phi^{k_{st}}}{1-\phi^2}.
$$

Therefore,

$$
\boxed{
\frac{\partial \ell}{\partial \sigma^2}
=
\frac{\alpha^2}{2(1-\phi^2)}
\sum_{s=0}^T
\sum_{t=0}^T
B_{st}\phi^{k_{st}}.
}
$$

The derivative with respect to $\tau^2$ is

$$
\frac{\partial (\Sigma_\theta)_{st}}{\partial \tau^2}
=
\mathbf{1}_{s=t}.
$$

Therefore,

$$
\boxed{
\frac{\partial \ell}{\partial \tau^2}
=
\frac{1}{2}
\sum_{t=0}^T B_{tt}.
}
$$

For $\phi$, we have

$$
\frac{\partial}{\partial \phi}
\left(
\frac{\phi^k}{1-\phi^2}
\right)
=
\frac{
k\phi^{k-1}(1-\phi^2)
+
2\phi^{k+1}
}{
(1-\phi^2)^2
}.
$$

Therefore,

$$
\frac{\partial (\Sigma_\theta)_{st}}{\partial \phi}
=
\alpha^2\sigma^2
\frac{
k_{st}\phi^{k_{st}-1}(1-\phi^2)
+
2\phi^{k_{st}+1}
}{
(1-\phi^2)^2
}.
$$

So

$$
\boxed{
\frac{\partial \ell}{\partial \phi}
=
\frac{\alpha^2\sigma^2}{2}
\sum_{s=0}^T
\sum_{t=0}^T
B_{st}
\frac{
k_{st}\phi^{k_{st}-1}(1-\phi^2)
+
2\phi^{k_{st}+1}
}{
(1-\phi^2)^2
}.
}
$$
For $k=|t-s|$,

$$
D_\phi(k)
=
\frac{\partial}{\partial \phi}
\left(
\frac{\phi^k}{1-\phi^2}
\right).
$$

If $k=0$, then

$$
D_\phi(0)
=
\frac{2\phi}{(1-\phi^2)^2}.
$$

If $k\geq 1$, then

$$
D_\phi(k)
=
\frac{
k\phi^{k-1}(1-\phi^2)
+
2\phi^{k+1}
}{
(1-\phi^2)^2
}.
$$

Therefore,

$$
\frac{\partial \ell}{\partial \phi}
=
\frac{\alpha^2\sigma^2}{2}
\sum_{s=0}^T
\sum_{t=0}^T
B_{st}D_\phi(|t-s|).
$$

# Likelihood Ratio

For two parameter values $\theta$ and $\theta'$,

$$
\frac{p(y_{0:T}\mid \theta')}{p(y_{0:T}\mid \theta)}
=
\sqrt{
\frac{|\Sigma_\theta|}{|\Sigma_{\theta'}|}
}
\exp
\left[
-\frac{1}{2}y^\top
\left(
\Sigma_{\theta'}^{-1}
-
\Sigma_\theta^{-1}
\right)y
\right].
$$

Equivalently,

$$
\boxed{
\Lambda(\theta'\mid\theta)
=
\sqrt{
\frac{|\Sigma_\theta|}{|\Sigma_{\theta'}|}
}
\exp\left[
\frac12 y^\top
\left(
\Sigma_\theta^{-1}
-
\Sigma_{\theta'}^{-1}
\right)y
\right].
}
$$

# Autodiff for Calculating the Score

In this model, the score is the gradient of the observed-data log-likelihood with respect to the model parameters.

For the 1D linear-Gaussian state-space model,

$$
\begin{cases}
x_t = \phi x_{t-1} + \epsilon_t, \\
y_t = \alpha x_t + \eta_t, \\
\epsilon_t \sim \mathcal{N}(0,\sigma^2), \\
\eta_t \sim \mathcal{N}(0,\tau^2),
\end{cases}
$$

the observed-data likelihood is

$$
p(y_{0:T}\mid \theta)
=
\int p(x_{0:T},y_{0:T}\mid \theta)\,dx_{0:T}.
$$

In the linear-Gaussian case, this integral can be computed exactly by the Kalman filter. Therefore,

$$
\ell(\theta;y_{0:T})
=
\log p(y_{0:T}\mid \theta)
$$

can be computed recursively by the Kalman filter.

The score is

$$
\nabla_\theta \ell(\theta;y_{0:T}).
$$

Instead of deriving the derivative of every Kalman recursion by hand, automatic differentiation differentiates through the computational graph of the Kalman likelihood.

---

# Kalman Likelihood as a Differentiable Program

The scalar Kalman likelihood recursion is:

$$
S_t = \alpha^2 P_{t|t-1}+\tau^2,
$$

$$
v_t = y_t-\alpha m_{t|t-1},
$$

$$
\ell_t
=
-\frac12
\left[
\log(2\pi)
+
\log S_t
+
\frac{v_t^2}{S_t}
\right].
$$

The total log-likelihood is

$$
\ell(\theta;y_{0:T})
=
\sum_{t=0}^T \ell_t.
$$

The Kalman update is

$$
K_t = \frac{\alpha P_{t|t-1}}{S_t},
$$

$$
m_{t|t}
=
m_{t|t-1}+K_t v_t,
$$

$$
P_{t|t}
=
(1-K_t\alpha)^2P_{t|t-1}
+
K_t^2\tau^2.
$$

The prediction step is

$$
m_{t+1|t}
=
\phi m_{t|t},
$$

$$
P_{t+1|t}
=
\phi^2P_{t|t}+\sigma^2.
$$

Since all these operations are differentiable, autodiff can compute

$$
\nabla_\theta \ell(\theta;y_{0:T})
$$

by backpropagating through the Kalman recursion.

---

# Why Autodiff Works Here

The log-likelihood is a composition of differentiable operations:

$$
\theta
\mapsto
(\phi,\alpha,\sigma^2,\tau^2)
\mapsto
(S_t,v_t,K_t,m_{t|t},P_{t|t})_{t=0}^T
\mapsto
\ell(\theta;y_{0:T}).
$$

Therefore, by the chain rule,

$$
\nabla_\theta \ell(\theta;y_{0:T})
$$

can be computed automatically.

This avoids manually writing sensitivity recursions such as

$$
\frac{\partial m_{t|t}}{\partial \theta_j},
\qquad
\frac{\partial P_{t|t}}{\partial \theta_j},
\qquad
\frac{\partial S_t}{\partial \theta_j},
\qquad
\frac{\partial v_t}{\partial \theta_j}.
$$

Autodiff computes these derivatives implicitly.

---

# Constrained Versus Unconstrained Parameters

Hamiltonian Monte Carlo should usually operate on unconstrained parameters.

A common parametrization is

$$
u_\phi \in \mathbb{R},
\qquad
u_{\sigma^2} \in \mathbb{R},
\qquad
u_{\tau^2} \in \mathbb{R},
$$

with

$$
\phi = \tanh(u_\phi),
$$

$$
\sigma^2 = \exp(u_{\sigma^2}),
$$

and

$$
\tau^2 = \exp(u_{\tau^2}).
$$

If $\alpha$ is fixed, for example

$$
\alpha=1,
$$

then the HMC parameter is

$$
u = (u_\phi,u_{\sigma^2},u_{\tau^2}).
$$

The log-likelihood becomes a function of the unconstrained parameters:

$$
\widetilde{\ell}(u;y)
=
\ell(\theta(u);y).
$$

The score needed by HMC is then

$$
\nabla_u \widetilde{\ell}(u;y).
$$

Autodiff computes this directly by differentiating through both:

1. the parameter transformation $u\mapsto \theta(u)$;
2. the Kalman likelihood recursion $\theta\mapsto \ell(\theta;y)$.

---

# Chain Rule Interpretation

If the score with respect to constrained parameters is

$$
\nabla_\theta \ell(\theta;y),
$$

then the score with respect to unconstrained parameters is

$$
\nabla_u \widetilde{\ell}(u;y)
=
J_\theta(u)^\top
\nabla_\theta \ell(\theta;y),
$$

where

$$
J_\theta(u)
=
\frac{\partial \theta(u)}{\partial u}.
$$

For example,

$$
\frac{\partial \phi}{\partial u_\phi}
=
1-\tanh^2(u_\phi)
=
1-\phi^2,
$$

$$
\frac{\partial \sigma^2}{\partial u_{\sigma^2}}
=
\sigma^2,
$$

and

$$
\frac{\partial \tau^2}{\partial u_{\tau^2}}
=
\tau^2.
$$

Therefore,

$$
\frac{\partial \widetilde{\ell}}{\partial u_\phi}
=
(1-\phi^2)
\frac{\partial \ell}{\partial \phi},
$$

$$
\frac{\partial \widetilde{\ell}}{\partial u_{\sigma^2}}
=
\sigma^2
\frac{\partial \ell}{\partial \sigma^2},
$$

and

$$
\frac{\partial \widetilde{\ell}}{\partial u_{\tau^2}}
=
\tau^2
\frac{\partial \ell}{\partial \tau^2}.
$$

Autodiff applies this chain rule automatically.

---

# Log Posterior for HMC

HMC targets the posterior, not just the likelihood.

The posterior density is

$$
p(u\mid y_{0:T})
\propto
p(y_{0:T}\mid \theta(u))p(u).
$$

Thus the log posterior is

$$
\log \pi(u)
=
\widetilde{\ell}(u;y)
+
\log p(u).
$$

The HMC score is

$$
\nabla_u \log \pi(u)
=
\nabla_u \widetilde{\ell}(u;y)
+
\nabla_u \log p(u).
$$

If priors are placed directly on the unconstrained parameters $u$, then no Jacobian correction is needed.

If priors are placed on the constrained parameters $\theta$, then the transformed posterior is

$$
\log \pi(u)
=
\ell(\theta(u);y)
+
\log p(\theta(u))
+
\log
\left|
\det
\frac{\partial \theta}{\partial u}
\right|.
$$

For

$$
\phi=\tanh(u_\phi),
\qquad
\sigma^2=\exp(u_{\sigma^2}),
\qquad
\tau^2=\exp(u_{\tau^2}),
$$

the log-Jacobian is

$$
\log
\left|
\det
\frac{\partial \theta}{\partial u}
\right|
=
\log(1-\phi^2)
+
u_{\sigma^2}
+
u_{\tau^2}.
$$
