from exptools.collections import namedarraytuple

# Some algorithms might not use the field "next_observation"
# For the unity of interface, it will still be provided
# NOTE: Under some cases, their will be no leading dim for each elements, so leading dims will be
# specified in the code
Trajectory = namedarraytuple("Trajectory", ["observation", "action", "reward", "done", "next_observation"])
