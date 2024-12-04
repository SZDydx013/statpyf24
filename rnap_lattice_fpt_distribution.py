#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
from lattice import Lattice


num_simulations = int(1e4)

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

first_passage_times = simulated_lattice.first_passage_time_list(
    num_simulations=num_simulations
)

# Calculate statistics
mean_first_passage_time = np.mean(first_passage_times)
std_dev_first_passage_time = np.std(first_passage_times)

print("First Passage Times Statistcs")
print(f"Sample Size:\t{len(first_passage_times)}")
print(f"Mean:\t\t\t{mean_first_passage_time:.2f}")
print(f"Standard Deviation:\t{std_dev_first_passage_time:.2f}")

# Plot histogram of first-passage times
plt.figure(figsize=(10, 6))
plt.hist(first_passage_times, bins=30, color='skyblue', edgecolor='black')
plt.xlabel("First-Passage Time")
plt.ylabel("Frequency")
plt.title(
    f"Histogram of First-Passage Times (Target at Site {
        simulated_lattice.target_site
    })")
plt.axvline(mean_first_passage_time, color='red', linestyle='--',
            label=f"Mean = {mean_first_passage_time:.2f}")
plt.legend()
plt.show()
