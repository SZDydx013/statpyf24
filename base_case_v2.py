"""

This version assumes the following:
    
    TF attaches at any site along the lattice with equal probability
    TF has no rate of detachment and will walk unitl it reaches the target site
    There are no RNAPs present to impede the path
    TF directional walk rate is not influenced by anything
    
"""


import numpy as np
import matplotlib.pyplot as plt


class base_TF:

    def __init__(self, lattice_size=100, target_position=50, rate_constant=1.0):
        
        self.lattice_size = lattice_size
        self.target_position = target_position
        self.rate_constant = rate_constant
        self.prng = np.random.default_rng()
        
    
    def simulate_random_walk(self, start_position):
        """Simulates the random walk for a single TF until it reaches the target position."""
        
        position = start_position
        time = 0.0
        trajectory = [(time, position)]

        while position != self.target_position:
            # Determine waiting time for the next step
            tau = self.prng.exponential(scale=1/self.rate_constant)
            time += tau

            # Randomly choose direction: -1 (left) or +1 (right) and stay wuthin lattice bounds
            step = self.prng.choice([-1, 1])
            position = max(0, min(self.lattice_size - 1, position + step))  

            trajectory.append((time, position))
        
        return time, trajectory


    def compute_fpt(self, simulations=100):
        """Computes the average FPT for each starting position across multipl simulations."""
        
        fpt_results = []
        
        for start in range(self.lattice_size):
            fpt_list = [self.simulate_random_walk(start)[0] for sim in range(simulations)]
            avg_fpt = np.mean(fpt_list)
            fpt_results.append(avg_fpt)
        
        return fpt_results


    def plot_fpt_vs_position(self, fpt_results):
        """Plots FPT vs Starting Position."""
        
        plt.figure(figsize=(10, 6))
        plt.plot(range(self.lattice_size), fpt_results, linestyle='-')
        plt.axhline(np.mean(fpt_results), color='red', linestyle='--', label=f"Mean = {(np.mean(fpt_results)):.2f}")
        plt.title('First Passage Time vs Starting Position')
        plt.xlabel('Starting Position')
        plt.ylabel('Average First Passage Time')
        plt.grid()
        plt.legend()
        plt.show()


    def plot_sample_walk(self, trajectory):
        """Plots a single random walk trajectory."""
        
        times, positions = zip(*trajectory)
        plt.figure(figsize=(10, 6))
        plt.plot(times, positions, linestyle='-')
        plt.title('Sample Random Walk')
        plt.xlabel('Time')
        plt.ylabel('Lattice Position')
        plt.axhline(self.target_position, color='r', linestyle='--', label='Target Position')
        plt.legend()
        plt.grid()
        plt.show()



# Initiate and run simulation 
tf_simulation = base_TF()
fpt_results = tf_simulation.compute_fpt(25)
print(f"Mean FPT: {(np.mean(fpt_results)):.2f}")


# Plot results
tf_simulation.plot_fpt_vs_position(fpt_results)
unused, sample_trajectory = tf_simulation.simulate_random_walk(start_position=0)
tf_simulation.plot_sample_walk(sample_trajectory)
