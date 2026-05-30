# HLW Macro State-Space Model

Inspired by Holston, Laubach, and Williams (2017). Jointly estimates the output gap
$x_t$, potential growth $g_t^*$, natural unemployment $u_t^*$, and neutral real
interest rate $r_t^*$ from four quarterly observed series.

---

## 1. Observables and Units

| Symbol | Series | Transform |
|---|---|---|
| $\Delta Y_t$ | Real GDP growth | $100 \times \Delta \log \texttt{GDPC1}$ |
| $\pi_t$ | Inflation | annualised, e.g. $400 \times \Delta \log \texttt{PCEPILFE}$ |
| $u_t$ | Unemployment rate | $\texttt{UNRATE}$, percent |
| $i_t$ | Nominal interest rate | e.g. federal funds rate, percent |

All units are in percentage points. The ex-ante real interest rate constructed from
data is $r_t = i_t - \pi_t$.

---

## 2. Parameters

| Parameter | Sign / Range | Interpretation |
|---|---|---|
| $\phi_1,\, \phi_2$ | — | IS-curve AR(1) and AR(2) coefficients |
| $\lambda_r$ | $< 0$ | IS-curve sensitivity to real rate gap |
| $c_g$ | $> 0$ | Loading of trend growth onto neutral rate |
| $\alpha_\pi$ | $(0, 1)$ | Inflation persistence in Phillips curve |
| $\beta_\pi$ | $> 0$ | Unemployment-gap coefficient in Phillips curve |
| $\gamma$ | $> 0$ | Output-gap coefficient in Okun's law |
| $\rho_i$ | $[0, 1)$ | Monetary policy inertia |
| $\psi_\pi$ | $> 1$ | Inflation response in Taylor rule (Taylor principle) |
| $\psi_x$ | $> 0$ | Output-gap response in Taylor rule |
| $\pi^*$ | fixed | Central bank inflation target |
| $\sigma_x,\, \sigma_g,\, \sigma_\zeta,\, \sigma_{u^*}$ | $> 0$ | Latent-state shock standard deviations |
| $\sigma_{\Delta Y},\, \sigma_\pi,\, \sigma_u,\, \sigma_i$ | $> 0$ | Observation noise standard deviations |

Total free parameters (with $\pi^*$ fixed): **18**.

---

## 3. Structural Form

This section presents the model equations in their economically interpretable form.

### 3.1 Latent Dynamics

**IS curve** (HLW specification, two-quarter average real rate gap):

$$
x_t
= \phi_1 x_{t-1} + \phi_2 x_{t-2}
+ \lambda_r \cdot \frac{(r_{t-1} - r_{t-1}^*) + (r_{t-2} - r_{t-2}^*)}{2}
+ \varepsilon_{x,t}
$$

where $r_{t-k} = i_{t-k} - \pi_{t-k}$ is the ex-ante real rate constructed from
observed data. The sign $\lambda_r < 0$ ensures that a real rate above its neutral
level depresses output.

**Trend output growth** (random walk):

$$
g_t^* = g_{t-1}^* + \varepsilon_{g,t}
$$

**Persistent non-growth component of the neutral rate** (random walk):

$$
\zeta_t = \zeta_{t-1} + \varepsilon_{\zeta,t}
$$

**Neutral real interest rate** (deterministic function of other states, no own shock):

$$
r_t^* = c_g\, g_t^* + \zeta_t
$$

**Natural unemployment rate / NAIRU** (random walk):

$$
u_t^* = u_{t-1}^* + \varepsilon_{u^*,t}
$$

Latent shocks are mutually independent:
$$
\varepsilon_{x,t} \sim \mathcal{N}(0,\sigma_x^2), \quad
\varepsilon_{g,t} \sim \mathcal{N}(0,\sigma_g^2), \quad
\varepsilon_{\zeta,t} \sim \mathcal{N}(0,\sigma_\zeta^2), \quad
\varepsilon_{u^*,t} \sim \mathcal{N}(0,\sigma_{u^*}^2)
$$

> **Note on $r_t^*$.** Because $r_t^* = c_g g_t^* + \zeta_t$ is an exact linear
> function of other states, its innovation is $c_g \varepsilon_{g,t} + \varepsilon_{\zeta,t}$
> — correlated with $g_t^*$ and $\zeta_t$ and with no additional variance of its own.
> The process noise matrix $Q$ is therefore not diagonal; see Section 4.3.

### 3.2 Observation Equations

**GDP growth identity:**

$$
\Delta Y_t = g_t^* + (x_t - x_{t-1}) + \varepsilon_{\Delta Y,t}
$$

Follows from $Y_t = Y_t^* + x_t$ with $\Delta Y_t^* \equiv g_t^*$.

**Phillips curve** (unemployment-gap, with adaptive inflation expectations):

Define the model-implied inflation signal:

$$
\tilde\pi_t
= \alpha_\pi\,\pi_{t-1} + (1-\alpha_\pi)\,\bar\pi_{t-2:4}
- \beta_\pi\,(u_{t-1} - u_{t-1}^*),
\qquad
\bar\pi_{t-2:4} = \tfrac{1}{3}(\pi_{t-2}+\pi_{t-3}+\pi_{t-4})
$$

Then the inflation observation is:

$$
\pi_t = \tilde\pi_t + \varepsilon_{\pi,t}
$$

When unemployment exceeds NAIRU the unemployment gap is positive, so $\beta_\pi > 0$
and the negative sign in front of $\beta_\pi$ correctly reduces inflationary pressure.
$\bar\pi_{t-2:4}$ is a backward-looking anchor. The convex combination
$\alpha_\pi \pi_{t-1} + (1-\alpha_\pi)\bar\pi_{t-2:4}$ controls the relative weight
on recent versus medium-term past inflation.

**Okun's law:**

$$
u_t = u_t^* - \gamma\, x_t + \varepsilon_{u,t}
$$

A positive output gap (boom) pushes unemployment below NAIRU.

**Inertial Taylor rule** (reacting to model-implied inflation $\tilde\pi_t$):

$$
i_t
= \rho_i\,i_{t-1}
+ (1-\rho_i)\!\left[r_t^* + \pi^* + \psi_\pi(\tilde\pi_t - \pi^*) + \psi_x\,x_t\right]
+ \varepsilon_{i,t}
$$

The neutral nominal rate is $r_t^* + \pi^*$ (Fisher equation). The rule satisfies the
Taylor principle when $\psi_\pi > 1$. Using $\tilde\pi_t$ (model-implied rather than
observed inflation) couples this equation to the Phillips curve; see Section 4.4 for
how this expands in the KF form.

Observation shocks are mutually independent and independent of latent shocks:
$$
\varepsilon_{\Delta Y,t} \sim \mathcal{N}(0,\sigma_{\Delta Y}^2), \quad
\varepsilon_{\pi,t}    \sim \mathcal{N}(0,\sigma_\pi^2), \quad
\varepsilon_{u,t}      \sim \mathcal{N}(0,\sigma_u^2), \quad
\varepsilon_{i,t}      \sim \mathcal{N}(0,\sigma_i^2)
$$

---

## 4. Kalman Filter Form

Standard form:

$$
s_t = A\,s_{t-1} + a_t + \eta_t,
\qquad \eta_t \sim \mathcal{N}(0,Q)
$$

$$
y_t = H\,s_t + d_t + \varepsilon_t,
\qquad \varepsilon_t \sim \mathcal{N}(0,R)
$$

$A$, $H$, $Q$, $R$ are time-invariant. $a_t$ and $d_t$ are time-varying offsets
computed from lagged observables before each predict/update step.

### 4.1 State Vector

$$
s_t =
\begin{bmatrix}
x_t \\ x_{t-1} \\ g_t^* \\ \zeta_t \\ r_t^* \\ r_{t-1}^* \\ u_t^* \\ u_{t-1}^*
\end{bmatrix}
\in \mathbb{R}^8
$$

The three auxiliary components $x_{t-1}$, $r_{t-1}^*$, $u_{t-1}^*$ carry lagged
values needed for the two-quarter IS-curve average and the Phillips curve
unemployment-gap term, keeping $A$ and $H$ time-invariant.

### 4.2 Transition Matrix $A$

Column order: $[x_t,\; x_{t-1},\; g_t^*,\; \zeta_t,\; r_t^*,\; r_{t-1}^*,\; u_t^*,\; u_{t-1}^*]$.

$$
A =
\begin{pmatrix}
\phi_1 & \phi_2 & 0   & 0 & -\tfrac{\lambda_r}{2} & -\tfrac{\lambda_r}{2} & 0 & 0 \\
1      & 0      & 0   & 0 & 0                     & 0                     & 0 & 0 \\
0      & 0      & 1   & 0 & 0                     & 0                     & 0 & 0 \\
0      & 0      & 0   & 1 & 0                     & 0                     & 0 & 0 \\
0      & 0      & c_g & 1 & 0                     & 0                     & 0 & 0 \\
0      & 0      & 0   & 0 & 1                     & 0                     & 0 & 0 \\
0      & 0      & 0   & 0 & 0                     & 0                     & 1 & 0 \\
0      & 0      & 0   & 0 & 0                     & 0                     & 1 & 0
\end{pmatrix}
$$

Row-by-row derivations:

| Row | State updated | Derivation |
|-----|--------------|------------|
| 1 | $x_t$ | IS curve; $-\lambda_r/2 > 0$ on $r_{t-1}^*$ and $r_{t-2}^*$ columns (subtracts natural rate contribution; real-rate-gap observable part goes in $a_t$). |
| 2 | $x_{t-1}$ | Copy of previous $x_t$: $A[1,0]=1$. |
| 3 | $g_t^*$ | Random walk identity. |
| 4 | $\zeta_t$ | Random walk identity. |
| 5 | $r_t^*$ | $r_t^* = c_g g_t^* + \zeta_t = c_g(g_{t-1}^*+\varepsilon_g) + (\zeta_{t-1}+\varepsilon_\zeta)$: $A[4,2]=c_g$, $A[4,3]=1$. |
| 6 | $r_{t-1}^*$ | Copy of previous $r_t^*$: $A[5,4]=1$. |
| 7 | $u_t^*$ | Random walk identity: $A[6,6]=1$. |
| 8 | $u_{t-1}^*$ | Copy of previous $u_t^*$: $A[7,6]=1$. |

### 4.3 Time-Varying Transition Offset $a_t$

Only row 1 (the IS curve) has a non-zero time-varying term, arising from the
observable part of the real rate gap:

$$
a_t =
\begin{pmatrix}
\lambda_r \cdot \dfrac{r_{t-1} + r_{t-2}}{2} \\[4pt]
0 \\ 0 \\ 0 \\ 0 \\ 0 \\ 0 \\ 0
\end{pmatrix},
\qquad
r_{t-k} = i_{t-k} - \pi_{t-k}
$$

Computed from $(i_{t-1}, \pi_{t-1}, i_{t-2}, \pi_{t-2})$ before the predict step.

### 4.4 Process Noise Covariance $Q$

The noise vector in the transition is
$\eta_t = [\varepsilon_x,\; 0,\; \varepsilon_g,\; \varepsilon_\zeta,\; c_g\varepsilon_g + \varepsilon_\zeta,\; 0,\; \varepsilon_{u^*},\; 0]^\top$.
The auxiliary rows (2, 6, 8) carry no noise. The $r_t^*$ row is driven by $\varepsilon_g$
and $\varepsilon_\zeta$, producing off-diagonal entries in the $(g^*, \zeta, r^*)$ block.

$$
Q =
\begin{pmatrix}
\sigma_x^2 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & \sigma_g^2 & 0 & c_g\sigma_g^2 & 0 & 0 & 0 \\
0 & 0 & 0 & \sigma_\zeta^2 & \sigma_\zeta^2 & 0 & 0 & 0 \\
0 & 0 & c_g\sigma_g^2 & \sigma_\zeta^2 & c_g^2\sigma_g^2+\sigma_\zeta^2 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & \sigma_{u^*}^2 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0
\end{pmatrix}
$$

$Q$ has rank 4 (driven by $\varepsilon_x$, $\varepsilon_g$, $\varepsilon_\zeta$, $\varepsilon_{u^*}$).
This is not an error: singular $Q$ is handled correctly by the standard Kalman recursion
as long as $R$ is positive definite.

### 4.5 Observation Matrix $H$

Expanding $\tilde\pi_t$ into $\pi_t$ and $i_t$, the only state-dependent term is
$+\beta_\pi u_{t-1}^*$ (column 7 in both rows). All lagged observable terms go into
$d_t$ below.

$$
H =
\begin{pmatrix}
1 & -1 & 1 & 0 & 0 & 0 & 0 & 0 \\[2pt]
0 & 0  & 0 & 0 & 0 & 0 & 0 & \beta_\pi \\[2pt]
-\gamma & 0 & 0 & 0 & 0 & 0 & 1 & 0 \\[2pt]
(1-\rho_i)\psi_x & 0 & 0 & 0 & (1-\rho_i) & 0 & 0 & (1-\rho_i)\psi_\pi\beta_\pi
\end{pmatrix}
$$

Row derivations:

| Row | Observation | State-dependent terms |
|-----|------------|----------------------|
| 1 | $\Delta Y_t$ | $x_t - x_{t-1} + g_t^*$: columns 0, 1, 2. |
| 2 | $\pi_t$ | $+\beta_\pi u_{t-1}^*$ (column 7); all lagged-$\pi$ and lagged-$u$ terms are in $d_t$. |
| 3 | $u_t$ | $u_t^* - \gamma x_t$: columns 6 and 0. |
| 4 | $i_t$ | $(1-\rho_i)\psi_x x_t$ (col 0) $+$ $(1-\rho_i)r_t^*$ (col 4) $+$ $(1-\rho_i)\psi_\pi\beta_\pi u_{t-1}^*$ (col 7, from expanding $\psi_\pi\tilde\pi_t$). |

### 4.6 Time-Varying Observation Offset $d_t$

$$
d_t =
\begin{pmatrix}
0 \\[4pt]
\alpha_\pi\,\pi_{t-1}
  + (1-\alpha_\pi)\,\bar\pi_{t-2:4}
  - \beta_\pi\,u_{t-1} \\[4pt]
0 \\[4pt]
\rho_i\,i_{t-1}
  + (1-\rho_i)\!\left[(1-\psi_\pi)\pi^*
    + \psi_\pi\!\left(\alpha_\pi\,\pi_{t-1}
      + (1-\alpha_\pi)\,\bar\pi_{t-2:4}
      - \beta_\pi\,u_{t-1}\right)\right]
\end{pmatrix}
$$

where $\bar\pi_{t-2:4} = \frac{1}{3}(\pi_{t-2}+\pi_{t-3}+\pi_{t-4})$.

The $i_t$ row simplifies by recognising that the bracketed expression is
$(1-\psi_\pi)\pi^* + \psi_\pi\,\tilde\pi_t^{\text{obs}}$, where
$\tilde\pi_t^{\text{obs}}$ is the observable part of $\tilde\pi_t$ (without
$+\beta_\pi u_{t-1}^*$).

**Data required at time $t$:**

| Lag | Quantities |
|-----|-----------|
| $t-1$ | $\pi_{t-1}$, $u_{t-1}$, $i_{t-1}$ |
| $t-2$ | $\pi_{t-2}$, $i_{t-2}$ (for $r_{t-2}$ in $a_t$) |
| $t-3$ | $\pi_{t-3}$ |
| $t-4$ | $\pi_{t-4}$ |

Construct $a_t$ from $(i_{t-1}, \pi_{t-1}, i_{t-2}, \pi_{t-2})$ at the predict step;
construct $d_t$ from $(\pi_{t-1:t-4}, u_{t-1}, i_{t-1})$ at the update step.

### 4.7 Observation Noise Covariance $R$

$$
R = \operatorname{diag}\!\left(\sigma_{\Delta Y}^2,\; \sigma_\pi^2,\; \sigma_u^2,\; \sigma_i^2\right)
$$

---

## 5. Implementation Notes

**Initialisation.** The random-walk components ($g_t^*, \zeta_t, u_t^*$) are
non-stationary, so the stationary Lyapunov equation $P_0 = AP_0A^\top + Q$ has no
finite solution for those dimensions. Use a diffuse prior for the random-walk
components and a stationary prior for $x_t$ (which has stable AR dynamics when
$|\phi_1 + \phi_2| < 1$).

**Singular $Q$.** Rows/columns for $x_{t-1}$, $r_{t-1}^*$, and $u_{t-1}^*$ are all
zero. The standard Kalman predict step $P_{t|t-1} = AP_{t-1}AP^\top + Q$ remains
well-defined; $P_{t|t-1}$ will be full-rank after the first update step if $R$ is
positive definite.

**Parameter constraints.** Enforce the following during MLE reparameterisation:

| Parameter | Transform |
|---|---|
| $\phi_1, \phi_2$ | constrain jointly to $|\phi_1 + \phi_2| < 1$ (stable AR) |
| $\lambda_r$ | $\lambda_r = -\exp(u_{\lambda_r})$ (negative) |
| $c_g, \beta_\pi, \gamma, \psi_x$ | $\exp(\cdot)$ (positive) |
| $\alpha_\pi$ | $\text{sigmoid}(\cdot)$ (in $(0,1)$) |
| $\psi_\pi$ | $1 + \exp(\cdot)$ (Taylor principle $> 1$) |
| $\rho_i$ | $\text{sigmoid}(\cdot)$ (in $[0,1)$) |
| $\sigma_{\cdot}$ | $\exp(\cdot)$ (positive) |

**Weak identification.** HLW (2017) document near-singular Hessians for $\sigma_g$,
$\sigma_\zeta$, and $c_g$, reflecting the difficulty of separating the two random-walk
drivers of $r_t^*$. Concentrated likelihood profiles or Bayesian priors on these three
parameters are advisable.

**Signal-to-noise ratios.** The ratio $\lambda = \sigma_{u^*} / \sigma_u$ (NAIRU
drift relative to Okun noise) governs how much the NAIRU estimate moves; similarly
$\sigma_g / \sigma_{\Delta Y}$ governs potential growth smoothness. These ratios are
often calibrated rather than freely estimated.