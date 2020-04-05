""" agents for bandit algorithms
"""
from girl.agents.base import AgentBase
from girl.concepts.spaces.int_box import IntBox
from girl.utils.quick_args import save__init__args

import torch
import numpy as np

class QTableAgent(AgentBase):
    """ NOTE: under the bandit setting, there is no condition on states, so this will be a bit
    different from normal RL settings
    """
    def __init__(self, **kwargs):
        save__init__args(locals())

    def initialize(self,
            observation_space, # This will not be used
            action_space: IntBox, # since the agent records a table, it need discrete action space
        ):
        assert not action_space.is_continuous, "How can you provide QTable with a continuous action space?"

        self.action_space = action_space
        assert len(action_space.shape) == 0, "In multi-armed bandit setting, the action is a single int"

        n_actions = (self.action_space.high - self.action_space.low)[0]
        # build a Q table with only action concerned
        self.q_table = torch.zeros((n_actions,), dtype= torch.float32)

class ActionCountAgent(QTableAgent):
    """ Considering oroginal Q table does not necessary store action_count as learning attribute.
    """
    def initialize(self, *args):
        super().initialize(*args)
        # This is a learned parameter, which is not manipulated in the agent
        self._action_count = torch.zeros_like(self.q_table, dtype= torch.uint32) # count actions
    @property
    def action_count_table(self):
        return self._action_count

class eGreedyAgent(ActionCountAgent):
    """ choose action based on epsilon-greedy strategy
    """
    def __init__(self, epsilon, **kwargs):
        super().__init__(epsilon= epsilon)

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
    def __init__(self, c, **kwargs):
        """ As UCB algorithm described, you need to provide a factor c
        """
        super().__init__(c= c)

    def reset(self, batch_size= 1):
        assert batch_size == 1, "UCB bandit algorithm support only batch size = 1 instead of {}".format(batch_size)
        self._t = torch.tensor(0., dtype= torch.float32) # recording the number of times self.step() is called
        super().reset(batch_size)

    @torch.no_grad()
    def step(self, observation):
        self._t += 1
        action_batch = list()
        for _ in range(self.batch_size):
            ucb_ = torch.sqrt(2 * torch.log(self._t) / self.action_count_table) * self.c
            ucb = self.q_table + ucb_
            action = torch.argmax(ucb)

            action_batch.append(action)
        return torch.stack(action_batch)
