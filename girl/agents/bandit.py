""" agents for bandit algorithms
"""
from girl.agents.base import AgentBase
from girl.concepts.spaces.int_box import IntBox
from girl.utils.quick_args import save__init__args

import torch
from torch.autograd import Variable
from torch.nn import functional as F
from torch import distributions as distr
import numpy as np

class QTableAgent(AgentBase):
    """ NOTE: under the bandit setting, there is no condition on states, so this will be a bit
    different from normal RL settings.
        And you need an attribute called `q_table` as torch tensor with shape (n,)
    """
    def initialize(self,
            observation_space, # This will not be used
            action_space: IntBox, # since the agent records a table, it need discrete action space
        ):
        assert not action_space.is_continuous, "How can you provide QTable with a continuous action space?"

        self.action_space = action_space
        assert len(action_space.shape) == 0, "In multi-armed bandit setting, the action is a single int"

        n_actions = self.action_space.high - self.action_space.low
        # build a Q table with only action concerned
        self.q_table = torch.zeros((n_actions,), dtype= torch.float32)

class ActionCountAgent(QTableAgent):
    """ Considering oroginal Q table does not necessary store action_count as learning attribute.
    """
    def initialize(self, *args, **kwargs):
        super().initialize(*args, **kwargs)
        # This is a learned parameter, which is not manipulated in the agent
        self._action_count = torch.zeros_like(self.q_table, dtype= torch.float32) # count actions
    @property
    def action_count_table(self):
        return self._action_count

class eGreedyAgent(ActionCountAgent):
    """ choose action based on epsilon-greedy strategy
    """
    def __init__(self, epsilon):
        save__init__args(locals())

    @torch.no_grad()
    def step(self, observation):
        action_batch = list()
        for _ in range(self.batch_size):
            if np.random.random() < self.epsilon:
                # random action
                action = self.action_space.sample()
                action = torch.from_numpy(action) # to make output class consistent
            else:
                # greedy action
                action = torch.argmax(self.q_table) # a scalar

            action_batch.append(action)
        return torch.stack(action_batch)

class ucbBanditAgent(ActionCountAgent):
    def __init__(self, c= 1.0):
        """ As UCB algorithm described, you need to provide a factor c
        """
        save__init__args(locals())

    def reset(self, batch_size= 1):
        assert batch_size == 1, "UCB bandit algorithm support only batch size = 1 instead of {}".format(batch_size)
        self._t = torch.tensor(0., dtype= torch.float32) # recording the number of times self.step() is called
        super().reset(batch_size)

    @torch.no_grad()
    def step(self, observation):
        self._t += 1

        action_batch = list()
        for _ in range(self.batch_size):

            # First make sure all actions are experienced once
            where_zero = np.where(self._action_count == 0)[0]
            if len(where_zero) > 0:
                action = torch.tensor(where_zero[0])
            else:
                ucb_ = torch.sqrt(2 * torch.log(self._t) / self.action_count_table)
                ucb = self.q_table + self.c * ucb_
                action = torch.argmax(ucb)

            action_batch.append(action)
        return torch.stack(action_batch)

class ThompsonAgent(QTableAgent):
    def __init__(self, prior: np.ndarray= None):
        """
        @ Args
            prior: a (n, 2) ndarray with beta priors. prior[:,0] is the first concentration
                If given, `n` has to be the same as number of valid actions
        """
        prior = torch.tensor(prior, dtype= torch.float32)
        save__init__args(locals())

    def initialize(self,
            observation_space,
            action_space: IntBox,
        ):
        if not self.prior is None:
            assert self.prior.shape[0] == (action_space.high - action_space.low), \
                "Wrong num of valid actions with distribution prior"
        else:
            self.prior = torch.ones(
                (action_space.high - action_space.low, 2),
                dtype= torch.float32,
            )
        bi_distr = distr.beta.Beta(self.prior[:,0], self.prior[:,1])
        self.q_table = bi_distr.sample()

    @torch.no_grad()
    def step(self, observation):
        bi_distr = distr.beta.Beta(self.prior[:,0], self.prior[:,1])
        self.q_table = bi_distr.sample()

        action_batch = list()
        for _ in range(self.batch_size):
            action_batch.append(self.q_table.argmax())
        
        return torch.stack(action_batch)

class GradientAgent(AgentBase):
    def __init__(self,
            random_init= False,
            beta= 1.0, # coefficient for likelyhood
            b: float= None, # if None, constantly update baseline; or keep baseline constant
        ):
        save__init__args(locals())
    
    def initialize(self,
            observation_space,
            action_space: IntBox,
        ):
        if not self.random_init:
            preference = torch.zeros((action_space.high - action_space.low, ))
        else:
            preference = torch.randn((action_space.high - action_space.low, ))
        self.preference = Variable(preference, requires_grad= True)

    def reset(self, batch_size):
        if self.b is None:
            self.baselines = list() # to record history rewards
        super().reset(batch_size= batch_size)

    def update_baselines(self, reward):
        """ NOTE: batch-wise reward
        """
        if self.b is None:
            self.baselines.append(reward)

    @property
    def baselines(self):
        """ A batch of baseline (mean of history reward)
        """
        if self.b is None:
            reward_history = torch.stack(self.baselines) # (T, B)
            baseline_t = torch.mean(reward_history, dim= 0,) # (B,)
        else:
            baseline_t = torch.ones((self.batch_size,), dtype= torch.float32) * self.b
        return baseline_t
    
    def parameters(self):
        """ justlike torch module
        """
        return [self.preference]

    def log_pi_table(self):
        """ Under Bandit setting, pi needs to output probability table for each action
        """
        return F.log_softmax(self.preference, dim= 0)

    @torch.no_grad()
    def step(self, observation):
        action_batch = list()
        for _ in range(self.batch_size):
            action_batch.append(self.preference.argmax())

        return torch.stack(action_batch)
