from girl.utils.quick_args import save__init__args

class SamplerBase:
    """ Sampler manages the environment and use the given agent to sample trajectories
    """
    def __init__(self,
            envCls,
            env_kwargs,
            **kwargs,
        ):
        save__init__args(locals())

    def initialize(self, agent):
        self.agent = agent

    def sample(self, itr_i):
        """ Sample by agent-env interaction, and return the trajectory.
        """
        raise NotImplementedError
