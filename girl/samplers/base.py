from girl.utils.quick_args import save__init__args
from girl.concepts.trajectory import Trajectory

class SamplerBase:
    """ Sampler manages the environment and use the given agent to sample trajectories
    """
    def __init__(self,
            EnvCls,
            env_kwargs,
            **kwargs,
        ):
        save__init__args(locals())

    def initialize(self, agent):
        self.agent = agent
        self.env = self.EnvCls(**self.env_kwargs)

    def sample(self, itr_i):
        """ Sample by agent-env interaction, and return the trajectory.
        @ Returns
            trajectory: Trajectory
        """
        raise NotImplementedError
