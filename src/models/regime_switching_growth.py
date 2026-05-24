# Similar to regime_switching.py
# slightly better suited for modeling relation between output gap and real GDP growth rate
'''
Model:

Latent process: 
    Regime: $$P(s_t = j | s_{t-1} = i) = P_{ij}$$
    Transition: $$x_t = \phi\, x_{t-1} + \sigma_{s_t}\, \epsilon_t, \quad \epsilon_t \sim N(0,1)$$
Observation:
    $$y_t = g^*_{s_t} + \mu\,(x_t - x_{t-1}) + \tau\, \eta_t, \quad \eta_t \sim N(0,1)$$

Augmented state: $a_t = [x_t,\, x_{t-1}]^\top$. The filter keeps one Gaussian per current regime and collapses after each update.
'''

from models.regime_switching import RegimeSwitchingSSM