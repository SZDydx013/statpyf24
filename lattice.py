#!/usr/bin/env python
"""
Obeys the following rules:

1) RNAPs can attach to the lattice at site 0 and will detach only at the end of the lattice. One RNAP being on the lattice does not stop another RNAP from attaching at site 0 (ie. multiple RNAPs can be present along the lattice at once). The rate at which new RNAPs attach to the lattice is a fixed value
2) RNAPs are unidirectional and will only move +1 until they reach the end
3) TFs can attach and detach from the lattice randomly at some defined rate. This means for each step a TF takes there is a chance it will also just remove itself from the lattice rather than proceed in some direction
4) 2 molecules cant occupy the same site at once. RNAP movement gets priority and if there is an RNAP in the way, the TF must move the other way rather than overlap with the RNAP
5) Only 1 TF will be on the lattice at once, though multiple RNAPs can. The total time taken for the TF to reach the target site is what's recorded. If the TF hops off the lattice and randomly attaches back on the lattice at another position the step counter does not reset. The count resets only once the TF has actually hit the target site
"""

import numpy as np


class Lattice:

    def __init__(
        self,
        lattice_length=100,
        target_site=None,
        rnap_attach_rate=0.2,
        rnap_move_rate=0.9,
        rnap_detach_rate=0.9,
        tf_attach_rate=0.01,
        tf_detach_rate=0.05,
        tf_move_rate=1.0,
        logging=False,
        step_limit=1e5
    ):
        # Lattice parameters
        self.lattice_length = lattice_length    # Length of the lattice
        # Target site will be in the middle of the lattice by default
        if target_site is not None:
            self.target_site = target_site
        else:
            self.target_site = lattice_length // 2

        # RNAP parameters
        # Probability of a new RNAP attaching to site 0 per step
        self.rnap_attach_rate = rnap_attach_rate
        # RNAP always moves forward (priority movement over TF)
        self.rnap_move_rate = rnap_move_rate
        # RNAP will detach at a fixed rate at the last site
        self.rnap_detach_rate = rnap_detach_rate

        # TF parameters
        # Probability of TF attaching at a random site per step
        self.tf_attach_rate = tf_attach_rate
        # Probability of TF detaching from the lattice per step
        self.tf_detach_rate = tf_detach_rate
        # TF can move randomly left (-1) or right (+1)
        self.tf_move_rate = tf_move_rate

        self.logging = logging

        self.step_limit = step_limit
        self.reset()

    def reset(self):
        # Lattice state variables
        self.lattice = np.zeros(self.lattice_length, dtype=int)
        self.lattice_age = 0
        self.tf_position = None
        self.rnap_positions = []

        # Lattice log varibles
        self.tf_path = []
        self.tf_detach_points = []
        self.tf_attach_points = []
        self.tf_block_points = []

    def simulate_step(self):
        self.lattice_age += 1
        # The detach → move → attach order was chosen to prevent the particles
        # from making multiple actions in a single step
        self.move_rnaps()
        self.attach_rnap()
        # RNAP takes priority over TF
        self.detach_tf()
        self.move_tf()
        self.attach_tf()

        return self.on_target()

    def simulate_to_target(self):
        self.reset()
        while not self.simulate_step():
            if self.lattice_age > self.step_limit:
                return None

        return self.lattice_age

    def site_empty(self, site_number):
        return self.lattice[site_number] == 0

    def rate_proceed(self, rate):
        return np.random.random() < rate

    def attach_rnap(self):
        # Attach RNAPs to site 0 with fixed probability, and only if site 0 is
        # empty
        if np.random.random() < self.rnap_attach_rate and self.site_empty(0):
            self.lattice[0] = 1
            self.rnap_positions.append(0)

    def move_rnaps(self):
        for rnap_number, position in enumerate(reversed(self.rnap_positions)):
            # Attempt to detach if RNAP is on the last site
            if (
                position == self.lattice_length - 1 and
                self.rate_proceed(self.rnap_detach_rate)
            ):
                self.lattice[position] = 0
                self.rnap_positions.pop(rnap_number)
                return
            # Move the RNAP at the rate if the next site is free
            new_position = position + 1
            if (
                self.site_empty(new_position) and
                self.rate_proceed(self.rnap_move_rate)
            ):
                self.lattice[position] = 0      # Empty the current position
                self.lattice[new_position] = 0  # Occupy the next position
                self.rnap_positions[rnap_number] = new_position

    def attach_tf(self):
        # Fail if the TF already exists or if the rate is too low
        if (
            self.tf_position is not None or
            not self.rate_proceed(self.tf_attach_rate)
        ):
            return
        empty_sites = np.where(self.lattice == 0)[0]
        if len(empty_sites) > 0:
            self.tf_position = np.random.choice(empty_sites)
            self.lattice[self.tf_position] == 2
            if self.logging:
                self.tf_attach_points.append(self.tf_position)
                self.tf_path.append(self.tf_position)

    def detach_tf(self):
        # Fail if the TF doesn't exist or the rate is too low
        if (
            self.tf_position is None or
            not self.rate_proceed(self.tf_detach_rate)
        ):
            return
        self.lattice[self.tf_position] = 0
        if self.logging:
            self.tf_detach_points.append(self.tf_position)
        self.tf_position = None

    def move_tf(self):
        # Fail if the TF doesn't exist or the rate is too low
        if (
            self.tf_position is None or
            not self.rate_proceed(self.tf_move_rate)
        ):
            return
        new_position = self.tf_position + np.random.choice([-1, 1])
        # Fail if the new position is out of the lattice
        if (
            new_position < 0 or
            new_position >= self.lattice_length
        ):
            return
        # Record if the new position is occupied and fail the move
        if not self.site_empty(new_position):
            if self.logging:
                self.tf_block_points.append(self.tf_position)
            return

        if self.logging:
            self.tf_path.append(new_position)

        self.lattice[self.tf_position] = 0
        self.lattice[new_position] = 2
        self.tf_position = new_position

    def on_target(self):
        return self.tf_position == self.target_site
