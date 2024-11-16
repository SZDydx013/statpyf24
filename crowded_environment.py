#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 15:20:54 2024

@author: akashpatel
"""
import numpy as np
import matplotlib.pyplot as plt


# Basic parameters
L = 100                     # Length of the lattice
target_site = L // 2        # Target site for the TF
num_simulations = 100       # Number of simulations to run
max_steps = 100000          # Maximum number of steps per simulation. If steps exceed this, TF has failed


# RNAP parameters
rnap_attach_rate = 0.2      # Probability of a new RNAP attaching to site 0 per step
rnap_move_rate = 1.0        # RNAP always moves forward by 1 (priority movement over TF)


# TF parameters
tf_attach_rate = 0.01       # Probability of TF attaching at a random site per step
tf_detach_rate = 0.05       # Probability of TF detaching from the lattice per step
tf_move_rate = 1.0          # TF can move randomly left (-1) or right (+1)


# Run simulation
first_passage_times = []

for sim in range(num_simulations):
    # Initialize lattice
    lattice = np.zeros(L, dtype=int)        # 0 = empty, 1 = RNAP, 2 = TF
    rnap_positions = []                     # List to track RNAP positions
    tf_position = None                      # TF position (None if not on the lattice)
    
    steps = 0
    
    while steps < max_steps:
        steps += 1
        
        # Step 1: RNAPs attach to site 0 with fixed probability
        if np.random.random() < rnap_attach_rate:
            if lattice[0] == 0:                         # Only attach if site 0 is empty
                lattice[0] = 1
                rnap_positions.append(0)
        
        # Step 2: RNAPs move forward if possible
        new_rnap_positions = []
        for pos in rnap_positions:
            if pos + 1 < L and lattice[pos + 1] == 0:   # Move to the next site
                lattice[pos] = 0                        # Clear current site
                lattice[pos + 1] = 1                    # Occupy next site
                new_rnap_positions.append(pos + 1)
            elif pos + 1 == L:                          # RNAP detaches at the end
                lattice[pos] = 0
            else:
                new_rnap_positions.append(pos)          # Stay if blocked
        
        rnap_positions = new_rnap_positions
        
        # Step 3: TF attaches randomly to an empty site
        if tf_position is None and np.random.random() < tf_attach_rate:
            tf_position = np.random.choice(np.where(lattice == 0)[0])           # Select a random site
            lattice[tf_position] = 2
        
        # Step 4: TF detaches with a fixed probability
        if tf_position is not None and np.random.random() < tf_detach_rate:
            lattice[tf_position] = 0                                            # Detach TF
            tf_position = None                                                  # Remove TF from the lattice
        
        # Step 5: TF moves randomly if on the lattice
        if tf_position is not None:
            move = np.random.choice([-1, 1])                                    # Random move in either direction
            new_position = tf_position + move
            
            if 0 <= new_position < L:                                           # Ensure new position is within bounds
                if lattice[new_position] == 0:                                  # Move if next site is empty
                    lattice[tf_position] = 0                                    # Clear current position
                    lattice[new_position] = 2                                   # Occupy new position
                    tf_position = new_position
                elif lattice[new_position] == 1:                                # Check if blocked by RNAP
                    if move == 1:                                               # If moving right, reverse
                        new_position = tf_position - 1
                    else:                                                       # If moving left, reverse
                        new_position = tf_position + 1
                    
                    if 0 <= new_position < L and lattice[new_position] == 0:    # Update TF position
                        lattice[tf_position] = 0
                        lattice[new_position] = 2
                        tf_position = new_position
        
        # Step 6: Check if TF reached the target site
        if tf_position == target_site:
            first_passage_times.append(steps)
            break


# Results
mean_time = np.mean(first_passage_times)
std_time = np.std(first_passage_times)

print(f"Mean First-Passage Time: {mean_time:.2f} steps")
print(f"Standard Deviation: {std_time:.2f} steps")


# Histogram of first-passage times
plt.figure(figsize=(10, 6))
plt.hist(first_passage_times, bins=20, color='lightblue', edgecolor='black')
plt.axvline(mean_time, color='red', linestyle='--', label=f"Mean = {mean_time:.2f}")
plt.xlabel("First-Passage Time (steps)")
plt.ylabel("Frequency")
plt.title(f"Histogram of Crowded Environment First-Passage Times (Target at Site {target_site})")
plt.legend()
plt.show()


# Visualization of a single simulation (TF path tracking)
lattice = np.zeros(L, dtype=int)        # Reset lattice
rnap_positions = []                     # Reset RNAP positions
tf_position = None                      # Reset TF position

positions = []                          # To track the movement of the TF
detachment_points = []                  # To record where the TF detaches
reattachment_points = []                # To record where the TF reattaches
block_points = []                       # To record where the TF was blocked by RNAP

steps = 0  

# Same simulation loop as above, only being run once to generate plot sample data
while steps < max_steps:
    steps += 1
    # Step 1: RNAPs attach to site 0
    if np.random.random() < rnap_attach_rate:
        if lattice[0] == 0:
            lattice[0] = 1
            rnap_positions.append(0)

    # Step 2: RNAPs move
    new_rnap_positions = []
    for pos in rnap_positions:
        if pos + 1 < L and lattice[pos + 1] == 0:
            lattice[pos] = 0
            lattice[pos + 1] = 1
            new_rnap_positions.append(pos + 1)
        elif pos + 1 == L:
            lattice[pos] = 0
        else:
            new_rnap_positions.append(pos)

    rnap_positions = new_rnap_positions

    # Step 3: TF attaches
    if tf_position is None and np.random.random() < tf_attach_rate:
        empty_sites = np.where(lattice == 0)[0]
        if len(empty_sites) > 0:
            tf_position = np.random.choice(empty_sites)
            lattice[tf_position] = 2
            reattachment_points.append(tf_position)         # Record reattachment lattice position

    # Step 4: TF detaches
    if tf_position is not None and np.random.random() < tf_detach_rate:
        lattice[tf_position] = 0
        detachment_points.append(tf_position)               # Record detachment lattice position
        tf_position = None

    # Step 5: TF moves
    if tf_position is not None:
        positions.append(tf_position)
        move = np.random.choice([-1, 1])
        new_position = tf_position + move

        if 0 <= new_position < L:
            if lattice[new_position] == 0:
                lattice[tf_position] = 0
                lattice[new_position] = 2
                tf_position = new_position
            elif lattice[new_position] == 1:
                block_points.append(tf_position)            # Record blockage lattice position
                new_position = tf_position - move
                if 0 <= new_position < L and lattice[new_position] == 0:
                    lattice[tf_position] = 0
                    lattice[new_position] = 2
                    tf_position = new_position

    if tf_position == target_site:
        positions.append(tf_position)
        break

# Print and Plot Results
print("\nData for a Single Simulation:\n")
print("Detachment Points (Lattice Positions):", detachment_points)
print("Reattachment Points (Lattice Positions):", reattachment_points)
print("Blockage Points (Lattice Positions):", block_points)
print(f"Steps Taken to Reach Target Site {target_site}: {len(positions)}")

plt.figure(figsize=(10, 6))
plt.plot(positions, label="TF Position")
plt.axhline(target_site, color='red', linestyle='--', label=f"Target Site = {target_site}")
plt.xlabel("Steps")
plt.ylabel("Lattice Position")
plt.title("TF Path to Target Site in a Crowded Lattice")
plt.legend()
plt.show()