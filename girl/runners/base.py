from girl.utils.quick_args import save__init__args
from girl.algos.base import AlgoBase
from girl.agents.base import AgentBase
from girl.samplers.base import SamplerBase

from exptools.logging import logger
from exptools.launching.affinity import set_gpu_from_visibles

import os
import torch

class RunnerBase:
    """ The controller that manages the running of the experiments.
    """
    def __init__(self,
            algo: AlgoBase,
            agent: AgentBase,
            sampler: SamplerBase,
            affinity,
            max_train_epochs: int,
            log_interval: int= 1,
            **kwargs,
        ):
        """
        @ Args
            max_train_epochs: The maximum number of training epoches,
                One epoch is one-time of calling algo.train()
            log_interval: The interval of actually logging into file
            affinity: Incase you run multiple experiment on one machine
        """
        self.algo = algo
        self.agent = agent
        self.sampler = sampler
        save__init__args(locals())

    def _startup(self):
        """ Setup the system and connect all components
        """
        # system setup
        logger.log(f"Runner{getattr(self, 'rank', '')} master Torch threads: "
            f"{torch.get_num_threads()}.")
        # view cuda configuration for this environment
        logger.log(f"Runner{getattr(self, 'rank', '')} CUDA_VISIBLE_DEVICES: "
            f"{os.environ.get('CUDA_VISIBLE_DEVICES', '')}.")
        set_gpu_from_visibles(self.affinity.get("cuda_idx", 0))

        # components setup
        self.agent.initialize(*self.sampler.env_spec())
        self.algo.initialize(self.agent)
        self.sampler.initialize(self.agent)

        # post components setup
        if torch.cuda.is_available():
            device = torch.device("cuda")
            self.agent.to(device= device)
        self.agent.sample_mode()

        # logging memory setup
        self._train_infos = {k: list() for k in self.algo.train_info_fields}
        self.epoch_i = 0

    def _get_epoch_snapshot(self, epoch_i):
        """ Collect all state needed for full checkpoint/snapshot of the training,
        including all model parameters and algorithm parameters
        """
        return dict(
            epoch_i= epoch_i,
            agent_state_dict= self.agent.state_dict(),
            algo_state_dict= self.algo.state_dict()
        )
    
    def _save_epoch_snapshot(self, epoch_i):
        """
        Calls the logger to save training checkpoint/snapshot (logger itself
        may or may not save, depending on mode selected).
        """
        logger.log("saving snapshot...")
        params = self._get_epoch_snapshot(epoch_i)
        logger.save_itr_params(epoch_i, params)
        logger.log("saved")

    def _store_train_info(self, epoch_i,
            env_info,
            train_info,
            extra_info,
        ):
        """ store train_info into attribute of self
        @ Args:
            env_info: a nested list(list(), ...) logging env_infos
            train_info: a namedtuple
            extra_info: a dict
        """
        for k, v in self._train_infos.items():
            new_v = getattr(train_info, k, [])
            v.extend(new_v if isinstance(new_v, list) else [new_v])

    def _log_dignostics(self, epoch_i):
        """ Call logger to dump all statistics to the file.
        NOTE: Due to calling logger.dump_tabular you need to call this implementation after you
        logging all your costomed information
        """
        self._save_epoch_snapshot(epoch_i)
        logger.record_tabular("Optim_epoch", epoch_i, epoch_i)

        for k, v in self._train_infos.items():
            if not k.startswith("_"):
                logger.record_tabular_misc_stat(k, v, epoch_i)
        self._train_infos = {k: list() for k in self._train_infos}

        logger.dump_tabular()

    def _load_snapshot(self, filename):
        """ A method to load parameters from snapshot and keep on training
        NOTE: filename has to be absolute path. And this has to be done after
        _startup
        """
        state_dict = torch.load(filename)
        logger.log("Loading snapshot from {}".format(filename))
        self.epoch_i = state_dict["epoch_i"]
        self.agent.load_state_dict(state_dict["agent_state_dict"])
        self.algo.load_state_dict(state_dict["algo_state_dict"])

    def _shutdown(self):
        """ Make sure all cleanup is done
        """
        pass

    def run(self, snapshot_path= None):
        """ The main loop of running experiment, and this is the only one that needed to be called
        by entrance script. If snapshot_path is given, you should load from the file path
        """
        self._startup()
        if snapshot_path is not None:
            self._load_snapshot(snapshot_path)
        
        # Finish startup start training
        while self.epoch_i < self.max_train_epochs:
            self.epoch_i += 1
            # Do the training procedure
            self.agent.sample_mode()
            env_info = self.sampler.sample(self.epoch_i)
            self.agent.train_mode()
            train_info, extra_info = self.algo.train(self.epoch_i, self.sampler.buffer_pyt, env_info)
            self.agent.sample_mode()

            # Do the logging, which is not part of the algorithm
            self._store_train_info(self.epoch_i,
                env_info,
                train_info,
                extra_info,
            )
            if self.epoch_i % self.log_interval == 0:
                self._log_dignostics(self.epoch_i)

        self._shutdown()
            
            