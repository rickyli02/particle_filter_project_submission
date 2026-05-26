# Rao-Blackwellized Particle Filter
# used for models where conditional distribution is partially closed form, such as regime switching models

# later, we will write block in an .ipynb notebook that test 
# 1. if RMSE, log likelihood variance, etc. are better with Rao-Blackwellized PF than naive PF for such models
# 2. if PMMH using RBPF is "better" than PMMH using naive PF

from .particle_filter import ParticleFilter
from utils import timer


class RaoBlackwellizedParticleFilter(ParticleFilter):
    def __init__(self, model=None, N_particles=10000, data=None, resample_method=None, seed=None):
        super().__init__(model, N_particles, data, resample_method, seed)
        # check that model is suitable for RBPF
        self.check_model()

    def check_model(self):
        # check that model is suitable for RBPF
        raise NotImplementedError

    @timer
    def run_filter(self):
        raise NotImplementedError

    def run_smoother(self):
        raise NotImplementedError