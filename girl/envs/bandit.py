from girl.utils.quick_args import save__init__args
from girl.envs.base import EnvBase, EnvInfo, EnvStep
from girl.concepts.spaces.int_box import IntBox

from collections import namedtuple
import numpy as np
from typing import List

BanditEnvInfoBase = namedtuple("BanditEnvInfoBase", ["V_star", "is_optimal"])
class BanditEnv(EnvBase):
    """ In the multi-armed bandit problem setting
    """
    def __init__(self,
            win_probs: List[float],
        ):
        """
        @ Args:
            win_probs: telling the winning probablilty of each arm, which also gives the number of arms
        """
        win_probs = np.array(win_probs)
        save__init__args(locals(), underscore=True)
        self._action_space = IntBox(0, len(self._win_probs))
        self._observation_space = IntBox(0, 1) # This serves no purpose, just to meet the interface
        self.BanditEnvInfo = namedtuple("BanditEnvInfo",
            [*BanditEnvInfoBase._fields] + ["arm{}".format(i) for i in range(len(win_probs))])

    def reset(self):
        return self._observation_space.null_value()

    def step(self, action):
        a = self._action_space.clamp(action)
        r = np.float32(np.random.binomial(1, self._win_probs[a]))
        o = self._observation_space.null_value()

        is_optimal = (self._win_probs[a] == np.amax(self._win_probs))
        action_count = np.zeros((len(self._win_probs),), dtype= np.int)
        action_count[a] = 1
        env_info = self.BanditEnvInfo(
                np.array(np.amax(self._win_probs)).astype(np.float32), # V_star
                is_optimal,
                *action_count
            )
        return EnvStep(
            observation= o.astype(np.float32),
            reward= r.astype(np.float32),
            done= False,
            env_info= env_info,
        )
