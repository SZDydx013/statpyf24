#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lattice import Lattice
from joblib import Parallel, delayed


def get_lattice_statistics(lattice_length, tf_initial_position):
    num_simulations = int(1e4)
    simulated_lattice = Lattice(
        lattice_length=int(lattice_length),
        rnap_attach_rate=0,  # Base case
        rnap_move_rate=1,
        rnap_detach_rate=1e6,
        tf_attach_rate=1e100,
        tf_move_rate=1,
        tf_detach_rate=0.02,
        step_limit=1e5,
    )

    first_passage_times = []

    for _ in range(num_simulations):
        simulated_lattice.reset()
        simulated_lattice.place_particle(1, tf_initial_position)
        time = simulated_lattice.simulate_to_target()
        if time is not None:
            first_passage_times.append(time)

    # Calculate statistics
    mean_first_passage_time = np.mean(first_passage_times)
    std_dev_first_passage_time = np.std(first_passage_times)

    return mean_first_passage_time, std_dev_first_passage_time


lattice_lengths = [1000]
lattice_positions = list(np.floor(np.linspace(0, 999, 100)))
lattice_positions.append(500)

for lattice_length in lattice_lengths:
    first_passage_time_stats = Parallel(n_jobs=-1)(
        delayed(get_lattice_statistics)(
            lattice_length, int(tf_initial_position))
        for tf_initial_position in lattice_positions
    )

    mean_times, std_dev_times = zip(*first_passage_time_stats)
    data = {'mean_times': mean_times, 'std_dev_times': std_dev_times}
    dataframe = pd.DataFrame(data=data)
    dataframe.to_csv(f'results/tf_position_{lattice_length}_results.csv')

    lower_bound_times, upper_bound_times = ([], [])
    for time_index, _ in enumerate(mean_times):
        lower_bound_times.append(
            mean_times[time_index] - std_dev_times[time_index])
        upper_bound_times.append(
            mean_times[time_index] + std_dev_times[time_index])

    print(mean_times)
    plt.plot(lattice_positions, mean_times, color='r', label='Mean')
    plt.fill_between(lattice_positions, lower_bound_times,
                     upper_bound_times, alpha=.3,
                     label='Standard deviation interval')
    plt.xlabel('TF initial position')
    plt.ylabel('First passage time')
    plt.gca().set_ylim(bottom=0)
    plt.legend()
    plt.show()
