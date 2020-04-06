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

    def train(self, epoch_i, trajs: Trajectory, env_info):
        """ Just to make sure the trajectory length and batch size are both 1
        """
        T, B = trajs.reward.shape[:2] # due to bandit setting, both of these are 1
        assert T == 1 and B == 1, "You sample too much for bandit algorithm T{}-B{}".format(T, B)

    def compute_metrics(self, epoch_i, trajs: Trajectory, env_info):
        """ Under bandit setting, B == T == 1
        """
        env_info = env_info[0][0]
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

    def train(self, epoch_i, trajs: Trajectory, env_info):
        super().train(epoch_i, trajs, env_info)
        action = trajs.action[0, 0]
        self.agent.action_count_table[action] += 1

        q = copy(self.agent.q_table[action])

        loss = trajs.reward[0, 0] - q
        self.agent.q_table[action] += 1./(self.agent.action_count_table[action]) * loss

        # Do some logging
        train_info = self.compute_metrics(epoch_i, trajs, env_info)
        extra_info = dict(
            action_count= self.agent.action_count_table,
        )

        return train_info, extra_info

class ThompsonAlgorithm(BanditAlgo):
    """ NOTE: Under Thompson sampling algorithm, the agent must be bandit.ThompsonAgent
    """
    def initialize(self, agent: ThompsonAgent):
        super().initialize(agent)

    def train(self, epoch_i, trajs: Trajectory, env_info):
        super().train(epoch_i, trajs, env_info)

        action = trajs.action[0, 0]
        reward = trajs.reward[0, 0]

        self.agent.prior[action, 0] += reward
        self.agent.prior[action, 1] += 1 - reward

        # Do some logging
        train_info = self.compute_metrics(epoch_i, trajs, env_info)
        extra_info = dict()
        return train_info, extra_info

class GradientBanditAlgo(BanditAlgo):
    train_info_fields = tuple(k for k in BanditTrainInfo._fields)

    def __init__(self, learning_rate= 1e-2):
        self.learning_rate = learning_rate

    def initialize(self, agent: GradientAgent):
        super().initialize(agent)

    def loss(self, trajs: Trajectory):
        """ calculate the loss which is used to compute gradient
        NOTE: some of the tensors here keep the batch dimension
        """
        action = trajs.action[0, 0]
        reward = trajs.reward[0] # (B,)
        baseline_table = self.agent.baseline_table

        indicator = torch.zeros_like(baseline_table)
        indicator[trajs.action[0,0]] = 1

        pi_loss = (reward - indicator) * self.agent.log_pi_table()[action]
        pi_loss = - pi_loss # for gradient ascent

        return pi_loss

    def train(self, epoch_i, trajs: Trajectory, env_info):
        super().train(epoch_i, trajs, env_info)

        self.agent.update_baseline(trajs.reward[0])

        # zero_grad
        params = self.agent.parameters()
        for param in params:
            param.grad.data.zero_()
        
        # compute loss (to target)
        pi_loss = self.loss(trajs)
        
        # compute gradient
        pi_loss.backward()

        # update parameters like gradient descent
        for param in params:
            param.data -= self.learning_rate * param.grad.data

        # Do some logging
        train_info = self.compute_metrics(epoch_i, trajs, env_info)
        extra_info = dict()
        return train_info, extra_info
    

    