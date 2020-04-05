""" For the most common 3 bandit algorithms, they follows almost the same procedure of optimizing
parameters.
"""
from girl.algos.base import AlgoBase
from girl.agents.bandit import ActionCountAgent
from girl.concepts.trajectory import Trajectory

from collections import namedtuple

TrainInfo = namedtuple("TrainInfo", ["loss", ])

class eGreedyBandit(AlgoBase):
    """ NOTE: Under bandit setting, the agent must be a ActionCountAgent
    """
    def initialize(self, agent: ActionCountAgent):
        super().initialize(agent)

    def train(self, epoch_i, trajs: Trajectory):
        T, B = trajs.reward.shape[:2] # due to bandit setting, both of these are 1
        assert T == 1 and B == 1, "You sample too much for bandit algorithm T{}-B{}".format(T, B)

        action = trajs.action[0, 0]
        self.agent.action_count_table[action] += 1

        loss = trajs.reward[0, 0] - self.agent.q_table[action]
        self.agent.q_table[action] += 1./(self.agent.action_count_table[action]) * loss

        # Do some logging
        train_info = TrainInfo(loss= loss.numpy())
        extra_info = dict(
            action_count= self.agent.action_count_table,
        )

        return train_info, extra_info

    