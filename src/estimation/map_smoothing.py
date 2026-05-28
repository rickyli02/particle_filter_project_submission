# Maximum A Posteriori Smoothing, i.e. over latent states
# Requires a closed form expression of p(x_{0:t} | y_{0:t}, \theta)
# The MAP estimator of the latent state is \hat{x}_{0:t} = argmax p(x_{0:t} | y_{0:t}, \theta)
# Note that this is a smoothing method, not a filtering method. It uses future data to estimate the past.
# The method is offline.
# Unlike the Kalman smoother (forward, backward Rauch-Tung-Striebel), the MAP smoother is not recursive and can handle non-Gaussian models.
# Optimization also makes it easier to impose constraints on latent state.
# In the case of linear Gaussian models, the MAP smoother should be equivalent to the Kalman smoother. Verify numerically.

# this is different from Parameter MAP Estimation, which is \theta_{MAP} = argmax p(\theta | y_{0:T} ) = argmax log g(\theta) + \sum_{t=0}^T log p(y_t | y_{0:t-1}, \theta)