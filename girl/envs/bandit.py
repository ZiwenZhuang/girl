from girl.utils.quick_args import save__init__args
from girl.envs.base import EnvBase, EnvInfo, EnvStep
from girl.spaces.int_box import IntBox

import numpy as np

class BanditEnv(EnvBase):
    """ In the multi-armed bandit problem setting
    """
    def __init__(self,
            win_probs: list(float),
        ):
        """
        @ Args:
            win_probs: telling the winning probablilty of each arm, which also gives the number of arms
        """
        save__init__args(locals(), underscore=True)
        self._action_space = IntBox(0, len(self._win_probs))
        self._observation_space = IntBox(0, 1) # This surves not purpose, just to meet the interface

    def reset(self):
        return self._observation_space.null_value()

    def step(self, action):
        a = self._action_space.clamp(action)
        r = np.random.binomial(1, self._win_probs[a])
        o = self._observation_space.null_value()

        return EnvStep(
            observation= o,
            reward= r,
            done= False,
            env_info= EnvInfo(),
        )
