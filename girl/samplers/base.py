from girl.utils.quick_args import save__init__args
from girl.agents.base import AgentBase
from girl.concepts.trajectory import Trajectory

from exptools.buffer import buffer_from_example, torchify_buffer

import numpy as np
import torch

class SamplerBase:
    """ Sampler takes control of all environments and use the given agent to sample trajectories
    """
    def __init__(self,
            EnvCls,
            env_kwargs,
            traj_len: int= 1,
            batch_size: int= 1,
            **kwargs,
        ):
        """
        @ Args:
            traj_len / T: maximum transitions in one sample trajectory. 
                If under some circumstance not reached, reset of them will be filled with zeros.
            batch_size / B: the batch size of one sample operation.
        """
        save__init__args(locals())

    def env_spec(self):
        """ To initialize an agent, you need to give it a information of observation and action.
        @ Returns
            observation_space: the observation space of the environment
            action_space: the action space of the environment
        """
        example_env = self.EnvCls(**self.env_kwargs)
        observation_space = example_env.observation_space
        action_space = example_env.action_space

        del example_env
        return observation_space, action_space

    def initialize(self, agent: AgentBase):
        self.agent = agent
        # construct environment instances
        self.envs = [self.EnvCls(**self.env_kwargs) for _ in range(self.batch_size)]

        # making an example to make trajectory batches
        o = self.envs[0].observation_space.sample()
        a = self.envs[0].action_space.sample()
        r = np.asarray(0, dtype= np.float32)
        d = np.asarray(0, dtype= np.uint8)
        o_ = o
        self.trajectory_example = Trajectory(o, a, r, d, o_)
        self.buffer_np = buffer_from_example(self.trajectory_example,
            leading_dims= (self.traj_len, self.batch_size)
        ) # with leading dim (T, B)
        self.buffer_pyt = torchify_buffer(self.buffer_np)
        # These two buffer share the same memory

        self.last_obs = np.stack([env.reset() for env in self.envs])
        self.agent.reset(batch_size= self.batch_size)

    def sample(self, epoch_i):
        """ Sample by agent-env interaction, and update trajectory buffer
        NOTE: You get trajectory via `buffer_pyt` or `buffer_np` attribute
        @ Returns
            env_infos: a nested list with leading dim (T, B) of environment returns
        """        
        self.buffer_np.observation[0] = self.last_obs
        env_infos = [[None for _ in range(self.batch_size)] for _ in range(self.traj_len)]
        # start collecting samples
        for t_i in range(self.traj_len):
            action = self.agent.step(torch.from_numpy(self.buffer_np.observation[t_i]))
            for b_i in range(len(self.envs)):
                o_, r, d, env_info = self.envs[b_i].step(action[b_i].numpy())
                self.buffer_np.next_observation[t_i, b_i] = o_
                self.buffer_np.reward[t_i, b_i] = r
                self.buffer_np.done[t_i, b_i] = d
                env_infos[t_i][b_i] = env_info
                # put to observation of next timestep if possible
                if t_i < self.traj_len-1:
                    self.buffer_np.observation[t_i+1, b_i] = o_
        # save observations for next sampling
        self.last_obs = self.buffer_np.observation[-1].copy()
        
        return env_infos
