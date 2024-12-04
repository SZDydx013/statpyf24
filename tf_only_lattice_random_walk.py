#!/usr/bin/env python
import matplotlib.pyplot as plt
from lattice import Lattice
'''
Get a random walk of a TF particle in a simple scenario.
'''

simulated_lattice = Lattice(
    lattice_length=100,
    rnap_attach_rate=0,  # There will be no RNAP in this simulation
    rnap_move_rate=1,
    rnap_detach_rate=1e6,
    tf_attach_rate=1e6,
    tf_move_rate=1,
    tf_detach_rate=0,  # TF will stay attached for the entire simulation
    step_limit=1e5,
)

# Sample random walk
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
