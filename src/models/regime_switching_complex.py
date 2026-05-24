# Regime switching model where regime transition probabilities depend on latent states
# use sigmoid function to model regime transition probabilities
# p(s_t = j | s_{t-1} = i, x_{t-1}) = sigmoid( f_ij (x_{t-1}) )

# because of the state dependence, RBPF and Kim smoother may not be applicable