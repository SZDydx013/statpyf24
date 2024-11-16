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

num_simulations = 100
simulated_lattice = Lattice(
    lattice_length=100,
    rnap_attach_rate=0,     # There will be no RNAPs in this
    tf_attach_rate=1,       # TF will attach on the first step
    tf_move_rate=1,         # TF will move on every step
    tf_detach_rate=0,       # TF will never detach
)

first_passage_times = []

for _ in range(num_simulations):
    first_passage_times.append(simulated_lattice.simulate_to_target())

# Calculate statistics
mean_first_passage_time = np.mean(first_passage_times)
std_dev_first_passage_time = np.std(first_passage_times)

print("First Passage Times Statistcs")
print(f"Mean:\t\t\t{mean_first_passage_time:.2f} steps")
print(f"Standard Deviation:\t{std_dev_first_passage_time:.2f} steps")

# Plot histogram of first-passage times
plt.figure(figsize=(10, 6))
plt.hist(first_passage_times, bins=30, color='skyblue', edgecolor='black')
plt.xlabel("First-Passage Time (steps)")
plt.ylabel("Frequency")
plt.title(
    f"Histogram of First-Passage Times (Target at Site {simulated_lattice.target_site})")
plt.axvline(mean_first_passage_time, color='red', linestyle='--',
            label=f"Mean = {mean_first_passage_time:.2f}")
plt.legend()
plt.show()

# Now a sample lattice
simulated_lattice.logging = True
simulated_lattice.simulate_to_target()

# plot the random walk
plt.figure(figsize=(10, 6))
plt.plot(simulated_lattice.tf_path, label="random walk of tf")
plt.axhline(simulated_lattice.target_site, color='red', linestyle='--',
            label=f"target site = {simulated_lattice.target_site}")
plt.xlabel("steps")
plt.ylabel("position on lattice")
print(simulated_lattice.tf_attach_points)
plt.title(f"sample random walk of tf (starting position: {
          simulated_lattice.tf_attach_points[0]}, total steps taken: {
          len(simulated_lattice.tf_path)})")
plt.legend()
plt.show()
