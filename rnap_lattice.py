#!/usr/bin/env python
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from lattice import Lattice

num_simulations = int(1e4)
simulated_lattice = Lattice(
    lattice_length=100,
    rnap_attach_rate=0.1,
    rnap_move_rate=1,
    rnap_detach_rate=1e6,
    tf_attach_rate=0.05,
    tf_move_rate=1,
    tf_detach_rate=0.02,
    step_limit=1e10,
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

visual_lattice = Lattice(
    lattice_length=10,
    rnap_attach_rate=0.2,
    rnap_move_rate=0.8,
    rnap_detach_rate=1,
    tf_attach_rate=0.2,
    tf_move_rate=0.8,
    tf_detach_rate=0.05,
)

(vid_fig, ani) = visual_lattice.visualization_video()
ffwriter = animation.FFMpegWriter(fps=10)
ani.save('animation.mp4', writer=ffwriter)
plt.show()
