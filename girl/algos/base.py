from girl.utils.quick_args import save__init__args

import torch
from torch import nn
from torch import optim

from collections import namedtuple

TrainInfo = namedtuple("TrainInfo", ["loss", "gradNorm"])

class AlgoBase:
    """ A basic implementation of the training algorithm
    """
    train_info_fields = tuple(f for f in TrainInfo._fields) # copy

    def __init__(self,
            loss_fn, # A torch.Tensor version loss function which should be tractable
            OptimCls= optim.Adam,
            learning_rate= 1e-5,
            weight_decay= 1e-2,
        ):
        save__init__args(locals())

    def initialize(self, agent):
        """ Register the agent which is to optimize, you may not inherit this implementation
        """
        self.agent = agent
        self.optim = self.OptimCls(
            self.model.parameters(),
            lr= self.learning_rate,
            weight_decay= self.weight_decay
        )

    def state_dict(self):
        """ summarize current state for snapshot
        """
        return dict()

    def load_state_dict(self, state):
        pass

    def train(self, itr_i, trajs):
        """ Perform one interation of optimization. Under most circumstance, it corresponding to
        one optim.step() call.
        @ Args:
            trajs: a list of (o, a, r, d, o') tuples
        @ returns:
            train_info: a namedtuple with numbered statistics
            extra_info: a dict depends on different problem, or algorithm
        """
        raise NotImplementedError
