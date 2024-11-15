#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

This version assumes the following:
    
    TF attaches at any site along the lattice with equal probability
    TF has no rate of detachment and will walk unitl it reaches the target site
    There are no RNAPs present to impede the path
    
"""

import numpy as np
import matplotlib.pyplot as plt

# Parameters
L = 100                     # Length of the lattice (DNA segment length)
target_site = L // 2        # Target site located at the middle of the lattice
num_simulations = 100       # Number of simulations to run


# Run simulation
first_passage_times = []
for sim in range(num_simulations):
    position = np.random.randint(0, L+1)     # Start TF at a random position on the lattice (equal probability for any position)
    steps = 0                                # Step counter for each simulation
    
    # Random walk until TF reaches the target or max steps are reached
    while position != target_site:
        step = np.random.choice([-1, 1])     # Random step: left (-1) or right (+1) along the lattice
        position += step                     # Update position
        position = max(0, min(L, position))  # Ensure TF stays within the lattice bounds
        steps += 1
    
    # Record first-passage time if the target was reached
    if position == target_site:
        first_passage_times.append(steps)
    else:
        continue


# Calculate statistics
first_passage_times = np.array(first_passage_times)
mean_first_passage_time = np.mean(first_passage_times)
std_first_passage_time = np.std(first_passage_times)

print(f"Mean First-Passage Time: {mean_first_passage_time:.2f} steps")
print(f"Standard Deviation: {std_first_passage_time:.2f} steps")


# Plot histogram of first-passage times
plt.figure(figsize=(10, 6))
plt.hist(first_passage_times, bins=30, color='skyblue', edgecolor='black')
plt.xlabel("First-Passage Time (steps)")
plt.ylabel("Frequency")
plt.title(f"Histogram of First-Passage Times (Target at Site {target_site})")
plt.axvline(mean_first_passage_time, color='red', linestyle='--', label=f"Mean = {mean_first_passage_time:.2f}")
plt.legend()
plt.show()


# Plot a sample random walk to visualize a search process for one simulation
position = np.random.randint(0, L+1)     # Random start position for the sample walk
positions = [position]

while position != target_site:
    step = np.random.choice([-1, 1])     # Random step: left (-1) or right (+1)
    position += step                     # Update position
    position = max(0, min(L, position))  # Ensure TF stays within lattice bounds
    positions.append(position)           # Add current position to array


# Plot the random walk
plt.figure(figsize=(10, 6))
plt.plot(positions, label="Random Walk of TF")
plt.axhline(target_site, color='red', linestyle='--', label=f"Target Site = {target_site}")
plt.xlabel("Steps")
plt.ylabel("Position on Lattice")
plt.title(f"Sample Random Walk of TF (Starting position: {positions[0]}, Total Steps Taken: {len(positions)})")
plt.legend()
plt.show()
