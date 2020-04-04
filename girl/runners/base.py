from girl.utils.quick_args import save__init__args
from exptools.logging import logger

import psutil
import time
import torch

class RunnerBase:
    """ The controller that manages the running of the experiments.
    """
    def __init__(self,
            algo,
            agent,
            sampler,
            **kwargs,
        ):
        save__init__args(locals())

    def startup(self):
        """ Setup the system and connect all components
        """
        # system setup
        logger.log(f"Runner {getattr(self, 'rank', '')} master Torch threads: "
            f"{torch.get_num_threads()}.")
        # view cuda configuration for this environment
        logger.log(f"Runner {getattr(self, 'rank', '')} CUDA_VISIBLE_DEVICES: "
            f"{os.environ['CUDA_VISIBLE_DEVICES']}.")
        if torch.cuda.is_available():
            device = torch.device("cuda")
            self.agent.to(device= device)

        # components setup
        self.agent.reset()
        self.algo.initialize(self.agent)
        self.sampler.initialize(self.agent)

        # logging setup
        self._train_infos = {k: list() for k in self.algo.train_info_fields}

    def get_itr_snapshot(self, itr_i):
        """ Collect all state needed for full checkpoint/snapshot of the training,
        including all model parameters and algorithm parameters
        """
        return dict(
            itr_i= itr_i,
            agent_state_dict= self.agent.state_dict(),
            algo_state_dict= self.algo.state_dict()
        )
    
    def save_itr_snapshot(self, itr_i):
        """
        Calls the logger to save training checkpoint/snapshot (logger itself
        may or may not save, depending on mode selected).
        """
        logger.log("saving snapshot...")
        params = self.get_epoch_snapshot(itr_i)
        logger.save_itr_params(itr_i, params)
        logger.log("saved")

    def store_train_info(self, itr_i, train_info, extra_info):
        """ store train_info into attribute of self
        @ Args:
            train_info: a namedtuple
            extra_info: a dict
        """
        for k, v in self._train_infos.items():
            new_v = getattr(train_info, k, [])
            v.extend(new_v if isinstance(new_v, list) else [new_v])

    def load_snapshot(self):
        """ A method to load parameters from snapshot and keep on sampling
        """
        pass

    def shutdown(self):
        """ Make sure all cleanup is done
        """
        pass

    def train(self):
        """ The main loop of make the training
        """