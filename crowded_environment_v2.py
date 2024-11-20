

import numpy as np
import matplotlib.pyplot as plt


class CrowdedEnvironment:
    
    def __init__(self, lattice_size=100, target_position=50, rnap_attach_rate=0.1, 
                 rnap_move_rate=1.0, tf_attach_rate=0.05, tf_detach_rate=0.02, tf_walk_rate=1.0):
        
        self.lattice_size = lattice_size
        self.target_position = target_position
        self.rnap_attach_rate = rnap_attach_rate
        self.rnap_move_rate = rnap_move_rate
        self.tf_attach_rate = tf_attach_rate
        self.tf_detach_rate = tf_detach_rate
        self.tf_walk_rate = tf_walk_rate
        
        self.prng = np.random.default_rng()
       
        
    def compute_fpt(self, simulations):
        """Computes the average FPT for each starting position across multiple simulations."""
        
        fpt_results = []
        
        for start in range(self.lattice_size):
            # Only keep the time (first element of the tuple)
            fpt_list = [self.simulate_tf_walk(start)[0] for sim in range(simulations)]
            avg_fpt = np.mean(fpt_list)
            fpt_results.append(avg_fpt)
        
        return fpt_results


    def simulate_tf_walk(self, start_position, time_threshold=5000):
        """Simulate a single TF walk with kinetic MC in a crowded environment."""
        
        lattice = np.zeros(self.lattice_size, dtype=int)                                  # 0 = empty, 1 = RNAP, 2 = TF
        rnap_positions = []
        tf_position = None
        time = 0.0
        trajectory = []
        
        # Place the TF on the lattice
        tf_position = start_position
        lattice[tf_position] = 2
        
        while tf_position != self.target_position:
            
            # Check if the total time has exceeded the threshold
            if time > time_threshold:
                return time_threshold, trajectory                                         # Stop and return the threshold time
    
            # Determine possible events and their rates
            events = []
            rates = []
            
            # RNAP attachment at site 0
            if lattice[0] == 0:
                events.append("rnap_attach")
                rates.append(self.rnap_attach_rate)
            
            # RNAP movement
            for pos in rnap_positions:
                if pos + 1 < self.lattice_size and lattice[pos + 1] == 0:
                    events.append(f"rnap_move_{pos}")
                    rates.append(self.rnap_move_rate)
            
            # TF detachment
            if tf_position is not None:
                events.append("tf_detach")
                rates.append(self.tf_detach_rate)
            
            # TF movement
            if tf_position is not None:
                if tf_position > 0 and lattice[tf_position - 1] == 0:                       # Move left
                    events.append("tf_move_left")
                    rates.append(self.tf_walk_rate)
                if tf_position < self.lattice_size - 1 and lattice[tf_position + 1] == 0:   # Move right
                    events.append("tf_move_right")
                    rates.append(self.tf_walk_rate)
            
            # Calculate cumulative rates
            total_rate = sum(rates)
            if total_rate == 0:
                break
            cumulative_rates = np.cumsum(rates) / total_rate
            
            # Determine the time until the next event
            tau = self.prng.exponential(scale=1/total_rate)
            time += tau
            
            # Determine which event occurs
            random_value = self.prng.uniform()
            chosen_event = events[np.searchsorted(cumulative_rates, random_value)]
            
            # Process the chosen event
            if chosen_event == "rnap_attach":
                rnap_positions.append(0)
                lattice[0] = 1
            elif "rnap_move" in chosen_event:
                rnap_index = int(chosen_event.split("_")[-1])
                lattice[rnap_index] = 0
                lattice[rnap_index + 1] = 1
                rnap_positions[rnap_positions.index(rnap_index)] += 1
            elif chosen_event == "tf_detach":
                lattice[tf_position] = 0
                tf_position = None
            elif chosen_event == "tf_move_left":
                lattice[tf_position] = 0
                tf_position -= 1
                lattice[tf_position] = 2
            elif chosen_event == "tf_move_right":
                lattice[tf_position] = 0
                tf_position += 1
                lattice[tf_position] = 2
            
            # If the TF is off the lattice, reattach randomly
            if tf_position is None:
                empty_sites = np.where(lattice == 0)[0]
                if len(empty_sites) > 0:
                    tf_position = self.prng.choice(empty_sites)
                    lattice[tf_position] = 2
            
            trajectory.append((time, tf_position))
        
        return time, trajectory


    def plot_fpt_vs_position(self, fpt_results):
        """Plot First Passage Time vs Starting Lattice Position."""
        
        plt.figure(figsize=(10, 6))
        plt.plot(range(self.lattice_size), fpt_results, linestyle='-', color='blue')
        plt.axhline(np.mean(fpt_results), color='red', linestyle='--', label=f"Mean = {(np.mean(fpt_results)):.2f}")
        plt.title('First Passage Time vs Starting Lattice Position')
        plt.xlabel('Starting Position on Lattice')
        plt.ylabel('Average First Passage Time')
        plt.grid()
        plt.legend()
        plt.show()


    def plot_sample_walk(self, trajectory):
        """Plot a single random walk trajectory."""
        
        times, positions = zip(*trajectory)
        plt.figure(figsize=(10, 6))
        plt.plot(times, positions, linestyle='-')
        plt.title('Sample Random Walk')
        plt.xlabel('Time')
        plt.ylabel('Lattice Position')
        plt.axhline(self.target_position, color='red', linestyle='--', label='Target Position')
        plt.legend()
        plt.grid()
        plt.show()



# Initialize and run simulation
sim = CrowdedEnvironment()
fpt_results = sim.compute_fpt(simulations=25)
print(f"Mean FPT: {(np.mean(fpt_results)):.2f}")


# Generate plots 
sim.plot_fpt_vs_position(fpt_results)
unused, sample_trajectory = sim.simulate_tf_walk(start_position=0)
sim.plot_sample_walk(sample_trajectory)
