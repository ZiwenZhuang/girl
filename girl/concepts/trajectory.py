from exptools.collections import namedarraytuple

# Some algorithms might not use the field "next_observation"
# For the unity of interface, it will still be provided
Trajectory = namedarraytuple("Trajectory", ["observation", "action", "reward", "done", "next_observation"])
