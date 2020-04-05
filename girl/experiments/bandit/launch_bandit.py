""" The main entrance of launching bandit experiment
"""
from exptools.collections import AttrDict
from exptools.launching.exp_launcher import run_experiments
from exptools.launching.variant import VariantLevel, make_variants, update_config
from exptools.launching.affinity import encode_affinity, quick_affinity_code

def make_default_config():
    return dict(
        env_kwargs= dict(
            win_probs= [0.9, 0.8, 0.7],
        ),
        solution= "eGreedy", # choose between "eGreedy", "ucb", "thompson", "gradientBandit" 
        agent_kwargs= dict(
            epsilon= 0.1,
            c= 1.0,
        ),
        algo_kwargs= dict(),
        sampler_kwargs= dict(
            traj_len= 1,
            batch_size= 1,
        ),
        runner_kwargs= dict(
            max_train_epochs= int(5e3),
        ),
    )

def main():
    # Either manually set the resources for the experiment:
    affinity_code = encode_affinity(
        n_cpu_core=4,
        n_gpu=1,
        contexts_per_gpu= 4,
        # hyperthread_offset=8,  # if auto-detect doesn't work, number of CPU cores
        # n_socket=1,  # if auto-detect doesn't work, can force (or force to 1)
        cpu_per_run=1,
    )
    # Or try an automatic one, but results may vary:
    # affinity_code = quick_affinity_code(n_parallel=None, use_gpu=True)

    default_config = make_default_config()

    # start building variants
    variant_levels = list()

    values = [
        [0.1,],
        [0.5,],
        [0.9,],
    ]
    dir_names = ["{}".format(*v) for v in values]
    keys = [("agent_kwargs", "epsilon"),] # each entry in the list is the string path to your config
    variant_levels.append(VariantLevel(keys, values, dir_names))

    # get all variants and their own log directory
    variants, log_dirs = make_variants(*variant_levels)
    for i, variant in enumerate(variants):
        variants[i] = update_config(default_config, variant)
    
    run_experiments(
        script= "girl/experiments/bandit.py",
        affinity_code= affinity_code,
        experiment_title= "Bandits",
        runs_per_setting= 1,
        variants= variants,
        log_dirs= log_dirs, # the directory under "${experiment title}"
        debug_mode= args.debug, # if greater than 0, the launcher will run one variant in this process)
    )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--debug', help= 'A common setting of whether to entering debug mode for remote attach',
        type= int, default= 0,
    )

    args = parser.parse_args()
    # setup for debugging if needed
    if args.debug > 0:
        # configuration for remote attach and debug
        import ptvsd
        import sys
        ip_address = ('0.0.0.0', 5050)
        print("Process: " + " ".join(sys.argv[:]))
        print("Is waiting for attach at address: %s:%d" % ip_address, flush= True)
        # Allow other computers to attach to ptvsd at this IP address and port.
        ptvsd.enable_attach(address=ip_address, redirect_output= True)
        # Pause the program until a remote debugger is attached
        ptvsd.wait_for_attach()
        print("Process attached, start running into experiment...", flush= True)
        ptvsd.break_into_debugger()

    main(args)