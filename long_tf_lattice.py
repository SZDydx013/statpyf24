#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
from lattice import Lattice
from joblib import Parallel, delayed


def get_lattice_statistics(lattice_length):
    num_simulations = int(1e5)
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

    first_passage_times = []

    for _ in range(num_simulations):
        simulated_lattice.reset()
        time = simulated_lattice.simulate_to_target()
        if time is not None:
            first_passage_times.append(time)

    # Calculate statistics
    mean_first_passage_time = np.mean(first_passage_times)
    std_dev_first_passage_time = np.std(first_passage_times)

    return mean_first_passage_time, std_dev_first_passage_time


lattice_length_powers = np.linspace(1, 3)
lattice_lengths = 10**lattice_length_powers

tf_movement_stats = Parallel(n_jobs=-1)(
    delayed(get_lattice_statistics)(lattice_length)
    for lattice_length in lattice_lengths
)

mean_times, std_dev_times = zip(*tf_movement_stats)

lower_bound_times, upper_bound_times = ([], [])

for time_index, _ in enumerate(mean_times):
    lower_bound_times.append(
        mean_times[time_index] - std_dev_times[time_index])
    upper_bound_times.append(
        mean_times[time_index] + std_dev_times[time_index])


plt.xscale('log')

plt.plot(lattice_lengths, mean_times, color='r', label='Mean')
plt.fill_between(lattice_lengths, lower_bound_times,
                 upper_bound_times, alpha=.3, label='Standard deviation interval')

plt.xlabel('Lattice Length')
plt.ylabel('First passage time')
plt.legend()
plt.show()
