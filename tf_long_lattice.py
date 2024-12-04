#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
from lattice import Lattice
from joblib import Parallel, delayed

num_simulations = int(1e2)


def get_lattice_statistics(lattice_length):
    simulated_lattice = Lattice(
        lattice_length=int(lattice_length),
        rnap_attach_rate=0,  # Base case
        rnap_move_rate=1,
        rnap_detach_rate=1e6,
        tf_attach_rate=0.05,
        tf_move_rate=1,
        tf_detach_rate=0.02,
        step_limit=1e5,
    )

    return simulated_lattice.first_passage_time_distribution(
        num_simulations=num_simulations
    )


lattice_length_powers = np.linspace(1, 3)
lattice_lengths = 10**lattice_length_powers

first_passage_time_stats = []

for lattice_length in lattice_lengths:
    first_passage_time_stats.append(
        get_lattice_statistics(lattice_length)
    )


mean_times, std_dev_times = zip(*first_passage_time_stats)

lower_bound_times, upper_bound_times = ([], [])

for time_index, _ in enumerate(mean_times):
    lower_bound_times.append(
        mean_times[time_index] - std_dev_times[time_index])
    upper_bound_times.append(
        mean_times[time_index] + std_dev_times[time_index])


plt.xscale('log')

# Plot the mean FPT time
plt.plot(lattice_lengths, mean_times, color='r', label='Mean')
# Plot the standard deviation interval of FPT
plt.fill_between(lattice_lengths, lower_bound_times,
                 upper_bound_times, alpha=.3,
                 label='Standard deviation interval'
                 )

plt.xlabel('Lattice Length')
plt.ylabel('First passage time')
plt.legend()
plt.show()
