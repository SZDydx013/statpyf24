#!/usr/bin/env python
import matplotlib.pyplot as plt
from lattice import Lattice
'''
Get how the initial point of TF affects the first passage time in a crowded scenario
'''

# Higher number means more accurate. 1e2 minimum, 1e3 recommended
# 1e3 takes 1 hour on a Ryzen 7640U. This is the most intensive simulation
num_simulations = int(1e1)

simulated_lattice = Lattice(
    lattice_length=100,
    rnap_attach_rate=0.2,
    rnap_move_rate=1,
    rnap_detach_rate=1e6,
    tf_attach_rate=1e6,
    tf_move_rate=1,
    tf_detach_rate=0.02,
    step_limit=1e5,
)

first_passage_time_stats = []

for starting_position in range(simulated_lattice.lattice_length):
    first_passage_time_stats.append(
        simulated_lattice.first_passage_time_distribution(
            num_simulations, starting_position
        )
    )

mean_times, std_dev_times = zip(*first_passage_time_stats)
lower_bound_times, upper_bound_times = ([], [])
for time_index, _ in enumerate(mean_times):
    lower_bound_times.append(
        mean_times[time_index] - std_dev_times[time_index])
    upper_bound_times.append(
        mean_times[time_index] + std_dev_times[time_index])

plt.plot(
    range(simulated_lattice.lattice_length),
    mean_times,
    color='r',
    label='Mean'
)
plt.fill_between(range(simulated_lattice.lattice_length), lower_bound_times,
                 upper_bound_times, alpha=.3,
                 label='Standard deviation interval')
plt.xlabel('TF initial position')
plt.ylabel('First passage time')
plt.gca().set_ylim(bottom=0)
plt.legend()
plt.show()
