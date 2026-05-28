import numpy as np
from models.linear_gaussian import LinearGaussianSSM


class LinearMacroSSM(LinearGaussianSSM):
    """
    Linear macro state-space model.

    Latent state  z_t = [x_t, g_t*, u_t*, pi_t^e, r_t*, x_{t-1}]' in R^6:
        x_t      — output gap
        g_t*     — potential GDP growth
        u_t*     — natural unemployment rate
        pi_t^e   — expected inflation
        r_t*     — neutral real interest rate
        x_{t-1}  — lagged output gap (augmentation to keep H constant)

    Transition:
        z_t = A z_{t-1} + a + eps_t,   eps_t ~ N(0, Q)

        A = [[rho_x, 0,     0,     0,        -lambda_r, 0],
             [0,     rho_g, 0,     0,         0,        0],
             [0,     0,     rho_u, 0,         0,        0],
             [0,     0,     0,     rho_pi,    0,        0],
             [0,     0,     0,     0,         rho_r,    0],
             [1,     0,     0,     0,         0,        0]]

        a = [0, (1-rho_g)g_bar, (1-rho_u)u_bar, (1-rho_pi)pi_bar, (1-rho_r)r_bar, 0]'

        Q = diag(sigma_x^2, sigma_g^2, sigma_u^2, sigma_pi^2, sigma_r^2, 0)

    Observation  y_t = [DeltaY_t, u_t, pi_t, i_t]' in R^4:
        y_t = H z_t + b_t + eta_t,   eta_t ~ N(0, R)

        H = [[1,            1, 0, 0,                      0,        -1],
             [-beta_u,      0, 1, 0,                      0,         0],
             [kappa,        0, 0, 1,                      0,         0],
             [(1-rho_i)*phi_x, 0, 0, (1-rho_i)*(1+phi_pi), (1-rho_i), 0]]

        b_t = [0, 0, 0, rho_i * i_{t-1} - (1-rho_i)*phi_pi*pi_bar]'  (time-varying)
    """

    def __init__(self, params, seed=None):
        raise NotImplementedError("LinearMacroSSM is not yet implemented.")
