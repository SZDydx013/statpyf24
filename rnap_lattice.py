#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
from lattice import Lattice

num_simulations = 100
simulated_lattice = Lattice(
    lattice_length=100,
    rnap_attach_rate=0.02,
    rnap_move_rate=0.8,
    rnap_detach_rate=1,
    tf_attach_rate=0.01,       # TF will attach on the first step
    tf_move_rate=1,         # TF will move on every step
    tf_detach_rate=0.05,       # TF will never detach
)

first_passage_times = []

for _ in range(num_simulations):
    time = simulated_lattice.simulate_to_target()
    if time is not None:
        first_passage_times.append(time)

# Calculate statistics
mean_first_passage_time = np.mean(first_passage_times)
std_dev_first_passage_time = np.std(first_passage_times)

print("First Passage Times Statistcs")
print(f"Sample Size:\t{len(first_passage_times)}")
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

sample_lattice = Lattice(
    lattice_length=10,
    rnap_attach_rate=0.2,
    rnap_move_rate=0.8,
    rnap_detach_rate=1,
    tf_attach_rate=0.2,
    tf_move_rate=0.8,
    tf_detach_rate=0.05,
)

'''
sample_lattice.simulate_to_target()

ani = animation.ArtistAnimation(
    plt.figure(), sample_lattice.visualization_image(), interval=50, repeat_delay=1000
)


plt.show()

print('assasasa')

'''
ani = sample_lattice.visualization_video()

plt.show()
'''
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
'''
