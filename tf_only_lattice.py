#!/usr/bin/env python
"""
This version assumes the following:

    TF attaches at any site along the lattice with equal probability
    TF has no rate of detachment and will walk unitl it reaches the target site
    There are no RNAPs present to impede the path

"""
import matplotlib.pyplot as plt
import numpy as np
from lattice import Lattice
from joblib import Parallel, delayed

num_simulations = int(1e5)
simulated_lattice = Lattice(
    lattice_length=100,
    rnap_attach_rate=0,     # There will be no RNAPs in this
    tf_attach_rate=1,       # TF will attach on the first step
    tf_move_rate=1,         # TF will move on every step
    tf_detach_rate=0,       # TF will never detach
)

first_passage_times = []

for _ in range(num_simulations):
    simulated_lattice.reset()
    passage_time = simulated_lattice.simulate_to_target()
    if passage_time is not None:
        first_passage_times.append(passage_time)

# Calculate statistics
mean_first_passage_time = np.mean(first_passage_times)
std_dev_first_passage_time = np.std(first_passage_times)

print("First Passage Times Statistcs")
print(f"Mean:\t\t\t{mean_first_passage_time:.2f}")
print(f"Standard Deviation:\t{std_dev_first_passage_time:.2f}")

# Plot histogram of first-passage times
plt.figure(figsize=(10, 6))
plt.hist(first_passage_times, bins=30, color='skyblue', edgecolor='black')
plt.xlabel("First-Passage Time")
plt.ylabel("Frequency")
plt.title(
    f"Histogram of First-Passage Times (Target at Site {simulated_lattice.target_site})")
plt.axvline(mean_first_passage_time, color='red', linestyle='--',
            label=f"Mean = {mean_first_passage_time:.2f}")
plt.legend()
plt.show()

# Walk Plot
simulated_lattice.reset()
simulated_lattice.logging = True
simulated_lattice.simulate_to_target()
times, positions = zip(*simulated_lattice.tf_path)
plt.figure(figsize=(10, 6))
plt.plot(times, positions, linestyle='-')
plt.title('Sample Random Walk')
plt.xlabel('Time')
plt.ylabel('Lattice Position')
plt.axhline(simulated_lattice.target_site, color='red',
            linestyle='--', label='Target Position')
plt.legend()
plt.grid()
plt.show()

# Time over position


def get_lattice_statistics(lattice_length, tf_initial_position):
    num_simulations = int(1e5)
    simulated_lattice = Lattice(
        lattice_length=int(lattice_length),
        rnap_attach_rate=0,
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
        simulated_lattice.place_particle(1, tf_initial_position)
        time = simulated_lattice.simulate_to_target()
        if time is not None:
            first_passage_times.append(time)

    # Calculate statistics
    mean_first_passage_time = np.mean(first_passage_times)
    std_dev_first_passage_time = np.std(first_passage_times)

    return mean_first_passage_time, std_dev_first_passage_time


lattice_length = 100
first_passage_time_stats = Parallel(n_jobs=-1)(
    delayed(get_lattice_statistics)(lattice_length, tf_initial_position)
    for tf_initial_position in range(lattice_length)
)

mean_times, std_dev_times = zip(*first_passage_time_stats)
lower_bound_times, upper_bound_times = ([], [])
for time_index, _ in enumerate(mean_times):
    lower_bound_times.append(
        mean_times[time_index] - std_dev_times[time_index])
    upper_bound_times.append(
        mean_times[time_index] + std_dev_times[time_index])

plt.plot(range(lattice_length), mean_times, color='r', label='Mean')
plt.fill_between(range(lattice_length), lower_bound_times,
                 upper_bound_times, alpha=.3,
                 label='Standard deviation interval')
plt.xlabel('TF initial position')
plt.ylabel('First passage time')
plt.gca().set_ylim(bottom=0)
plt.legend()
plt.show()
