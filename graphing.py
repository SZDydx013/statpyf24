#!/usr/bin/env python
'''
Not for release.
This is to graph results without restarting the simulation.
'''
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

lattice_positions = list(np.floor(np.linspace(0, 999, 100)))
midpoint = len(lattice_positions)//2
print(midpoint)
lattice_positions = lattice_positions[0:midpoint] + \
    [500] + lattice_positions[midpoint:]
dataframe = pd.read_csv('results/tf_position_1000_results.csv')
mean_times = dataframe['mean_times']
std_dev_times = dataframe['std_dev_times']

lower_bound_times, upper_bound_times = ([], [])
for time_index, _ in enumerate(mean_times):
    lower_bound_times.append(
        mean_times[time_index] - std_dev_times[time_index])
    upper_bound_times.append(
        mean_times[time_index] + std_dev_times[time_index])

plt.plot(lattice_positions, mean_times, color='r', label='Mean')
plt.fill_between(lattice_positions, lower_bound_times,
                 upper_bound_times, alpha=.3,
                 label='Standard deviation interval')
plt.xlabel('TF initial position')
plt.ylabel('First passage time')
plt.gca().set_ylim(bottom=0)
plt.legend()
plt.show()
