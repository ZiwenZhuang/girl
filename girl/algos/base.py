from girl.utils.quick_args import save__init__args
from girl.concepts.trajectory import Trajectory

import torch
from torch import nn
from torch import optim

from collections import namedtuple

TrainInfo = namedtuple("TrainInfo", ["loss"])

class AlgoBase:
    """ A basic implementation of the training algorithm
    """
    train_info_fields = tuple(f for f in TrainInfo._fields) # copy

    def __init__(self, **kwargs):
        save__init__args(locals())

    def initialize(self, agent):
        """ Register the agent which is to optimize, you may not inherit this implementation
        """
        self.agent = agent
        # if it is not a gradient based algorithm, there is no way to add torch optimizer

    def state_dict(self):
        """ summarize current state for snapshot
        """
        return dict()

    def load_state_dict(self, state):
        pass

    def train(self, epoch_i, trajs: Trajectory):
        """ Perform one interation of optimization. Under most circumstance, it corresponding to
        one optim.step() call.
        @ Args:
            trajs: a trajectory with leading dims (T, B)
        @ returns:
            train_info: a namedtuple with numbered statistics
            extra_info: a dict depends on different problem, or algorithm
        """
        raise NotImplementedError
