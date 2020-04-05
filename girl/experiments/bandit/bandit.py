"""
"""
from girl.agents.bandit import eGreedyAgent, ucbBanditAgent
from girl.algos.bandit import eGreedyBandit
from girl.envs.bandit import BanditEnv
from girl.samplers.base import SamplerBase
from girl.runners.base import RunnerBase

from exptools.launching.variant import load_variant
from exptools.launching.affinity import affinity_from_code
from exptools.logging.context import logger_context
import sys, os, json

def main(affinity_code, log_dir, run_id, *args):
    affinity = affinity_from_code(affinity_code)
    config = load_variant(log_dir)
    
    if config["solution"] == "eGreedy":
        agent_kwargs = {k: config["agent_kwargs"][k] for k in ('epsilon',)}
        agent = eGreedyAgent(**agent_kwargs)
        algo = eGreedyBandit(**config["algo_kwargs"])
    elif config["solution"] == "ucb":
        agent_kwargs = {k: config["agent_kwargs"][k] for k in ('c',)}
        agent = ucbBanditAgent(**agent_kwargs)
        algo = eGreedyBandit(**config["algo_kwargs"])
    else:
        raise NotImplementedError("Solution {} has not been implemented".format(config["solution"]))

    sampler = SamplerBase(**config["sampler_kwargs"])
    runner = RunnerBase(
        algo= algo, agent= agent, sampler= sampler,
        affinity= affinity,
        **config["runner_kwargs"]
    )

    name = "Bandits"
    with logger_context(log_dir, run_id, name, config):
        runner.run()

if __name__ == "__main__":
    # The main function name has to be "main" or "build_and_train" (conpatible with rlpyt)
    main(*sys.argv[1:]) # the argument will be put as follows:
        # ${affinity_code} ${log_dir} ${run_id} ${*args}