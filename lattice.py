#!/usr/bin/env python
"""
Obeys the following rules:

1) RNAPs can attach to the lattice at site 0 and will detach only at the end of the lattice. One RNAP being on the lattice does not stop another RNAP from attaching at site 0 (ie. multiple RNAPs can be present along the lattice at once). The rate at which new RNAPs attach to the lattice is a fixed value
2) RNAPs are unidirectional and will only move +1 until they reach the end
3) TFs can attach and detach from the lattice randomly at some defined rate. This means for each step a TF takes there is a chance it will also just remove itself from the lattice rather than proceed in some direction
4) 2 molecules cant occupy the same site at once. If there is an RNAP in the way, the TF must move the other way rather than overlap with the RNAP. If the TF is in the way, the RNAP cannot move unti; the TF moves.
5) Only 1 TF will be on the lattice at once, though multiple RNAPs can. The total time taken for the TF to reach the target site is recorded. If the TF hops off the lattice and randomly attaches back on the lattice at another position the timer does not reset. The time resets only once the TF has actually hit the target site.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation
from joblib import Parallel, delayed


class Lattice:

    def __init__(
        self,
        lattice_length=100,
        target_site=None,
        rnap_attach_rate=0.2,
        rnap_move_rate=1,
        rnap_detach_rate=1,
        tf_attach_rate=0.01,
        tf_detach_rate=0.005,
        tf_move_rate=1.0,
        logging=False,
        step_limit=1e6
    ):
        # Lattice parameters
        self.lattice_length = lattice_length    # Length of the lattice
        # Target site will be in the middle of the lattice by default
        if target_site is not None:
            self.target_site = target_site
        else:
            self.target_site = lattice_length // 2

        # RNAP parameters
        # Rate of which RNAP attaches to the start site
        self.rnap_attach_rate = rnap_attach_rate
        # Rate of which each RNAP moves +1
        self.rnap_move_rate = rnap_move_rate
        # Rate of which RNAP detaches from the end site
        self.rnap_detach_rate = rnap_detach_rate

        # TF parameters
        # Rate of which TF will attach to any site if there are no TF particles
        self.tf_attach_rate = tf_attach_rate
        # Rate of which TF will detach randomly
        self.tf_detach_rate = tf_detach_rate
        # Rate of which TF will move either direction
        self.tf_move_rate = tf_move_rate

        # Simulation parameters
        # Used to track the paths of particles. Turn off to save memory
        self.logging = logging
        # Used to avoid stalling the simulation.
        self.step_limit = step_limit

        # Random number generator
        self.prng = np.random.default_rng()

        self.reset()

    def reset(self):
        '''
        Remove all RNAP and TF from the lattice and reset its age
        '''
        # Lattice state variables
        # 0 = empty, 1 = RNAP, 2 = TF

        # Clean up the lattice
        self.lattice = np.zeros(self.lattice_length, dtype=int)
        # Reset lattice age
        self.lattice_age = 0.0
        # Reset simulation step count
        self.step_count = 0
        # Clean up TF tracking from the lattice
        self.tf_position = None
        # Clean up RNAP tracking from the lattice
        self.rnap_positions = []
        # Clean up event tracking from the lattice
        self.events = []

        # Lattice log varibles
        self.lattice_history = []
        self.tf_path = []
        self.tf_detach_points = []
        self.tf_attach_points = []
        self.tf_block_points = []

    def simulate_step(self):
        '''
        Compute the time until the next event, and then change the lattice
        based on the event.
        '''
        self.step_count += 1
        total_rate = self.collect_rates()
        # Determine the time until the next event
        self.lattice_age += self.prng.exponential(scale=1/total_rate)

        # Determine which event occurs
        self.execute_event(total_rate)

        if self.logging:
            self.tf_path.append((self.lattice_age, self.tf_position))
            self.lattice_history.append((self.lattice_age, self.lattice))

        return self.on_target()

    def simulate_to_target(self):
        '''
        Compute the time until the TF reaches the target site.
        This is done by stepping the simulation until the TF is on the
        target site.
        '''
        if self.on_target():
            return self.lattice_age
        while not self.simulate_step():
            if self.step_count > self.step_limit:
                print("Warning: TF never reached target")
                return None

        return self.lattice_age

    def get_single_first_passage_time(self, initial_TF_position=None):
        # Avoid parallel simulations from interfering with each other
        # By creating a new lattice for a full simulation
        temporary_lattice = Lattice(
            lattice_length=self.lattice_length,
            target_site=self.target_site,
            rnap_attach_rate=self.rnap_attach_rate,
            rnap_move_rate=self.rnap_move_rate,
            rnap_detach_rate=self.rnap_detach_rate,
            tf_attach_rate=self.tf_attach_rate,
            tf_detach_rate=self.tf_detach_rate,
            tf_move_rate=self.tf_move_rate,
            logging=self.logging,
            step_limit=self.step_limit,
        )

        # Insert a TF at a given point if given
        if initial_TF_position is not None:
            if (
                initial_TF_position < 0 or
                initial_TF_position >= temporary_lattice.lattice_length
            ):
                raise Exception("Initial TF position is outside lattice")
            temporary_lattice.place_particle(1, initial_TF_position)

        return temporary_lattice.simulate_to_target()

    def first_passage_time_list(
        self,
        num_simulations=1e2,
        initial_TF_position=None,
    ):
        first_passage_times = []

        # Simulate lattice many times
        first_passage_times = Parallel(n_jobs=-1)(
            delayed(self.get_single_first_passage_time)(initial_TF_position)
            for _ in range(num_simulations)
        )

        # Clear none values from the first passage times
        valid_first_passage_times = [
            i for i in first_passage_times if i is not None]

        return valid_first_passage_times

    def first_passage_time_distribution(
            self,
            num_simulations=1e2,
            initial_TF_position=None,
    ):
        first_passage_times = self.first_passage_time_list(
            num_simulations=num_simulations,
            initial_TF_position=initial_TF_position,
        )

        # Calculate statistics
        mean_first_passage_time = np.mean(first_passage_times)
        std_dev_first_passage_time = np.std(first_passage_times)

        return mean_first_passage_time, std_dev_first_passage_time

    def site_empty(self, site_number):
        '''
        Check if the given site is empty.
        This is used to check if particles can move to the site
        '''
        if site_number < 0 or site_number >= self.lattice_length:
            # Effectively the same as the particle cannot move out of the
            # lattice.
            return False  # If the site is out of bounds, it is "occupied"
        return self.lattice[site_number] == 0

    def collect_rates(self):
        '''
        Check the lattice state, observing where the TF and RNAPs are.

        This function will return the total rate and set the event rates class
        variable.
        '''
        self.events = []

        # RNAP attachment at site 0
        if self.site_empty(0):
            self.events.append(
                {"type": "rnap_attach", "rate": self.rnap_attach_rate}
            )

        # RNAP movement and detachment
        for rnap_index, position in enumerate(self.rnap_positions):
            new_pos_right = position + 1
            new_pos_left = position - 1

            # If the next site is empty, give the RNAP a chance to move either left or right
            if self.site_empty(new_pos_right):
                self.events.append({
                    "type": "rnap_move_right",
                    "rate": self.rnap_move_rate,
                    "rnap_pos": position,
                    "rnap_index": rnap_index,
                })
            if self.site_empty(new_pos_left):
                self.events.append({
                    "type": "rnap_move_left",
                    "rate": self.rnap_move_rate,
                    "rnap_pos": position,
                    "rnap_index": rnap_index,
                })

            # If RNAP is at the end of the lattice, give it a chance to detach
            if new_pos_right >= len(self.lattice):
                self.events.append({
                    "type": "rnap_detach",
                    "rate": self.rnap_detach_rate,
                    "rnap_pos": position,
                    "rnap_index": rnap_index,
                })

        if self.tf_position is None:  # TF attachment if it doesn't exist
            empty_sites = np.where(self.lattice == 0)[0]
            if len(empty_sites) > 0:
                self.events.append({
                    "type": "tf_attach",
                    "rate": self.tf_attach_rate,
                    "possible_sites": empty_sites,
                })
        else:  # TF movement and detachment if it exists
            if self.site_empty(self.tf_position - 1):  # Left movement
                self.events.append(
                    {"type": "tf_left", "rate": self.tf_move_rate}
                )
            if self.site_empty(self.tf_position + 1):  # Right movement
                self.events.append(
                    {"type": "tf_right", "rate": self.tf_move_rate}
                )
            # TF detachment
            self.events.append(
                {"type": "tf_detach", "rate": self.tf_detach_rate}
            )

        # Calculate total rate by adding all of the event rates
        total_rate = 0
        for event in self.events:
            total_rate += event["rate"]
        if total_rate == 0:
            print(self.lattice)
            # This should never happen unless rates were set improperly
            raise Exception("The lattice cannot do anything")

        # Calculate cumulative rates by adding the previous rates.
        # This is to allow for choosing of events
        prev_cumsum = 0
        for event in self.events:
            event["cum_rate"] = event["rate"] + prev_cumsum
            prev_cumsum += event["cum_rate"]

        return total_rate

    def execute_event(self, total_rate):
        '''
        From a list of events, choose what occurs based on probability
        '''
        chosen_event = None
        random_number = self.prng.uniform() * total_rate

        # Choose a random event from list of events
        for event in self.events:
            if random_number < event['cum_rate']:
                chosen_event = event
                break

        if chosen_event is None:
            # This will only happen if the random number generator
            # is not working and managed to generate a value above 1
            raise Exception('No even was chosen')

        match event['type']:
            case 'rnap_attach':
                # Attach an RNAP to the beginning of the lattice
                self.rnap_positions.append(0)
                self.lattice[0] = 1
            case 'rnap_move_left':
                # Move the RNAP to the left
                position = event['rnap_pos']
                new_position = event['rnap_pos'] - 1
                # Change the RNAP tracker information to match lattice state
                self.rnap_positions[event['rnap_index']] -= 1
                self.lattice[position] = 0
                self.lattice[new_position] = 1
            case 'rnap_move_right':
                # Move the RNAP to the right
                position = event['rnap_pos']
                new_position = event['rnap_pos'] + 1
                # Change the RNAP tracker information to match lattice state
                self.rnap_positions[event['rnap_index']] += 1
                self.lattice[position] = 0
                self.lattice[new_position] = 1
            case 'rnap_detach':
                # Detach RNAP at the end of the lattice
                self.rnap_positions.pop(event['rnap_index'])
                self.lattice[event['rnap_pos']] = 0
            case 'tf_attach':
                # Attach a TF particle at any unoccupied point in the lattice
                self.tf_position = self.prng.choice(event['possible_sites'])
                self.lattice[self.tf_position] = 2
            case 'tf_left':
                # Move the TF particle left
                self.lattice[self.tf_position] = 0
                self.tf_position -= 1
                self.lattice[self.tf_position] = 2
            case 'tf_right':
                # Move the TF particle right
                self.lattice[self.tf_position] = 0
                self.tf_position += 1
                self.lattice[self.tf_position] = 2
            case 'tf_detach':
                # Detach the TF particle randomly
                self.lattice[self.tf_position] = 0
                self.tf_position = None
            case _:
                # Should never be seen unless code was changed
                raise Exception(f'{event["type"]} is not a valid event')

    def on_target(self):
        '''
        Check if the TF is on the target site.
        '''
        return self.tf_position == self.target_site

    def place_particle(self, particle, position):
        '''
        Allow for easily manipulation of the lattice.
        Do not change the lattice using class variables as
        particle trackers and lattice state trackers are independent
        without proper function usage.
        '''
        if not (0 <= particle <= 2):
            # 0 = empty, 1 = TF, 2 = RNAP
            raise Exception(f"{particle} is not a valid particle")
        if not (0 <= position <= len(self.lattice)):
            raise Exception(f"{position} is outside of the lattice")

        # Clear the existing site and the associated particle tracker
        match self.lattice[position]:
            case 1:
                self.tf_position = None
            case 2:
                for rnap_index, rnap_positon in enumerate(self.rnap_positions):
                    if rnap_positon == position:
                        self.rnap_positions.pop[rnap_index]
        self.lattice[position] = 0

        # Place the particle and update the particle tracker
        match particle:
            case 1:
                if self.tf_position is not None:
                    raise Exception('There cannot be more than one TF')
                self.tf_position = position
            case 2:
                self.rnap_positions.append(position)
        self.lattice[position] = particle

    def remove_particle(self, position):
        '''
        Remove a particle at a lattice position
        '''
        self.place_particle(0, position)

    def site_drawing(self, ax, x, y, inside_color='w'):
        '''
        Draw a rectangle representing a site.
        '''
        ax.add_patch(mpatches.Rectangle(
            (x, y), 1, 1, color='k'))
        ax.add_patch(mpatches.Rectangle(
            (x+0.05, y+0.05), 0.9, 0.9, color=inside_color))

    def tf_drawing(self, ax, x, y):
        '''
        Draw a triangle representing a TF
        '''
        ax.add_patch(mpatches.CirclePolygon(
            (x+0.5, y+0.5), radius=0.4, color='g', resolution=3
        ))

    def rnap_drawing(self, ax, x, y):
        '''
        Draw a hexagon representing an RNAP
        '''
        ax.add_patch(mpatches.CirclePolygon(
            (x+0.5, y+0.5), radius=0.4, color='purple',
            resolution=6
        ))

    def visualization_image(self, time=None):
        '''
        Generate a visualization of the lattice
        '''
        if time is None:
            # Multiple images of the same lattice step may be generated.
            # This is to override the time label.
            time = self.lattice_age

        plt.axes()
        ax = plt.gca()
        plt.axis('off')
        ax.set_xlim(0, self.lattice_length)
        ax.set_ylim(0, 1.2)
        plt.text(0, 1.1, f"Time: {round(time, 1)}")

        for site_number in range(self.lattice_length):
            if site_number == self.target_site:
                site_color = 'orange'  # Make target site orange
            else:
                site_color = 'w'  # Make the site white otherwise
            # Draw the site
            self.site_drawing(ax, site_number, 0, site_color)
            match self.lattice[site_number]:
                case 1:
                    # Draw a TF particle if it is present on the site
                    self.rnap_drawing(ax, site_number, 0)
                case 2:
                    # Draw a RNAP particle if it is present on the site
                    self.tf_drawing(ax, site_number, 0)

        return [ax]

    def visualization_video(self):
        # Generate a video from a full simulation of the lattice
        self.reset()
        ims = []
        visualization_time = 0
        vid_fig = plt.figure()
        vid_fig.set_size_inches(self.lattice_length, 1.2)
        vid_fig.set_dpi(300)
        while not self.simulate_step():
            while visualization_time < self.lattice_age:
                ims.append(self.visualization_image(visualization_time))
                visualization_time += 0.1
            if self.step_count > self.step_limit:
                return None
        ims.append(self.visualization_image(visualization_time))
        return (vid_fig, animation.ArtistAnimation(
            vid_fig, ims, interval=100, repeat_delay=1000))
