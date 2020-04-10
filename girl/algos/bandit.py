""" For the most common 3 bandit algorithms, they follows almost the same procedure of optimizing
parameters.
"""
from girl.algos.base import AlgoBase
from girl.agents.bandit import ActionCountAgent, ThompsonAgent, GradientAgent
from girl.concepts.trajectory import Trajectory

from collections import namedtuple
from copy import copy
import numpy as np
import torch
from torch import functional as F

BanditTrainInfo = namedtuple("BanditTrainInfo", ["regret", "total_regret", "optimal_rate"])

class BanditAlgo(AlgoBase):
    """ To count regret and check bandit settings
    This class is not functionally, you have to chose one of its child class
    """
    train_info_fields = tuple(k for k in BanditTrainInfo._fields)

    def __init__(self):
        self.total_regret = np.zeros(tuple())
        self.optimal_counts = np.zeros(tuple())

    def train(self, epoch_i, trajs: Trajectory, env_infos):
        """ Just to make sure the trajectory length and batch size are both 1
        """
        T, B = trajs.reward.shape[:2] # due to bandit setting, both of these are 1
        assert T == 1 and B == 1, "You sample too much for bandit algorithm T{}-B{}".format(T, B)

    def compute_metrics(self, epoch_i, trajs: Trajectory, env_infos):
        """ Under bandit setting, B == T == 1
        """
        env_info = env_infos[0][0]
        regret = env_info.V_star - trajs.reward[0, 0].numpy()
        self.total_regret += regret

        if env_info.is_optimal:
            self.optimal_counts += 1
        optimal_rate = self.optimal_counts / epoch_i

        return BanditTrainInfo(
            regret= regret,
            total_regret= np.copy(self.total_regret),
            optimal_rate= optimal_rate,
        )

class eGreedyBandit(BanditAlgo):
    """ NOTE: Under bandit setting, the agent must be a ActionCountAgent
    """
    def initialize(self, agent: ActionCountAgent):
        super().initialize(agent)

    def train(self, epoch_i, trajs: Trajectory, env_infos):
        super().train(epoch_i, trajs, env_infos)
        action = trajs.action[0, 0]
        self.agent.action_count_table[action] += 1

        q = copy(self.agent.q_table[action])

        loss = trajs.reward[0, 0] - q
        self.agent.q_table[action] += 1./(self.agent.action_count_table[action]) * loss

        # Do some logging
        train_info = self.compute_metrics(epoch_i, trajs, env_infos)
        extra_info = dict(
            action_count= self.agent.action_count_table,
        )

        return train_info, extra_info

class ThompsonAlgorithm(BanditAlgo):
    """ NOTE: Under Thompson sampling algorithm, the agent must be bandit.ThompsonAgent
    """
    def initialize(self, agent: ThompsonAgent):
        super().initialize(agent)

    def train(self, epoch_i, trajs: Trajectory, env_infos):
        super().train(epoch_i, trajs, env_infos)

        action = trajs.action[0, 0]
        reward = trajs.reward[0, 0]

        self.agent.prior[action, 0] += reward
        self.agent.prior[action, 1] += 1 - reward

        # Do some logging
        train_info = self.compute_metrics(epoch_i, trajs, env_infos)
        extra_info = dict()
        return train_info, extra_info

class GradientBanditAlgo(BanditAlgo):
    train_info_fields = tuple(k for k in BanditTrainInfo._fields)

    def __init__(self, learning_rate= 1e-2):
        super().__init__()
        self.learning_rate = learning_rate

    def initialize(self, agent: GradientAgent):
        super().initialize(agent)
        self.agent_params = self.agent.parameters()

    def loss(self, trajs: Trajectory):
        """ calculate the loss which is used to compute gradient
        NOTE: some of the tensors here keep the batch dimension
        """
        action = trajs.action[0, 0]
        reward = trajs.reward[0] # (B,)
        baselines = self.agent.baselines[0]

        pi_loss = (reward - baselines) * self.agent.log_pi_table()[action]
        pi_loss = - pi_loss # for gradient ascent

        return pi_loss

    def train(self, epoch_i, trajs: Trajectory, env_infos):
        super().train(epoch_i, trajs, env_infos)

        self.agent.update_baselines(trajs.reward[0])

        # zero_grad
        for param in self.agent_params:
            if not param.grad is None:
                param.grad.data.zero_()
        
        # compute loss (to target)
        pi_loss = self.loss(trajs)
        
        # compute gradient
        pi_loss.backward()

        # update parameters like gradient descent
        for param in self.agent_params:
            param.data -= self.learning_rate * param.grad.data

        # Do some logging
        train_info = self.compute_metrics(epoch_i, trajs, env_infos)
        extra_info = dict()
        return train_info, extra_info
    

    