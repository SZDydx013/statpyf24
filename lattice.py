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
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation


class Lattice:

    def __init__(
        self,
        lattice_length=100,
        target_site=None,
        rnap_attach_rate=0.2,
        rnap_move_rate=0.9,
        rnap_detach_rate=1,
        tf_attach_rate=0.01,
        tf_detach_rate=0.004,
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
        # RNAP always moves forward (+1)
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

        # Simulation parameters
        self.logging = logging
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
        self.lattice = np.zeros(self.lattice_length, dtype=int)
        self.lattice_age = 0.0
        self.step_count = 0
        self.tf_position = None
        self.rnap_positions = []
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
        '''
        if self.on_target():
            return self.lattice_age
        while not self.simulate_step():
            if self.step_count > self.step_limit:
                print("Warning: TF never reached target")
                return None

        return self.lattice_age

    def site_empty(self, site_number):
        '''
        Check if the given site is empty.
        '''
        if site_number < 0 or site_number >= self.lattice_length:
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
            new_position = position + 1

            if self.site_empty(new_position):
                self.events.append({
                    "type": "rnap_move",
                    "rate": self.rnap_move_rate,
                    "rnap_pos": position,
                    "rnap_index": rnap_index,
                })

            if new_position >= len(self.lattice):
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

        # Calculate total rate
        total_rate = 0
        for event in self.events:
            total_rate += event["rate"]
        if total_rate == 0:
            print(self.lattice)
            raise Exception("The lattice cannot do anything")

        # Calculate cumulative rates
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

        for event in self.events:
            if random_number < event['cum_rate']:
                chosen_event = event
                break

        if chosen_event is None:
            raise Exception('No even was chosen')

        match event['type']:
            case 'rnap_attach':
                self.rnap_positions.append(0)
                self.lattice[0] = 1
            case 'rnap_move':
                position = event['rnap_pos']
                new_position = event['rnap_pos'] + 1

                self.rnap_positions[event['rnap_index']] += 1
                self.lattice[position] = 0
                self.lattice[new_position] = 1
            case 'rnap_detach':
                self.rnap_positions.pop(event['rnap_index'])
                self.lattice[event['rnap_pos']] = 0
            case 'tf_attach':
                self.tf_position = self.prng.choice(event['possible_sites'])
                self.lattice[self.tf_position] = 2
            case 'tf_left':
                self.lattice[self.tf_position] = 0
                self.tf_position -= 1
                self.lattice[self.tf_position] = 2
            case 'tf_right':
                self.lattice[self.tf_position] = 0
                self.tf_position += 1
                self.lattice[self.tf_position] = 2
            case 'tf_detach':
                self.lattice[self.tf_position] = 0
                self.tf_position = None
            case _:
                raise Exception(f'{event["type"]} is not a valid event')

    def on_target(self):
        '''
        Check if the TF is on the target site.
        '''
        return self.tf_position == self.target_site

    def place_particle(self, particle, position):
        if not (0 <= particle <= 2):
            raise Exception(f"{particle} is not a valid particle")
        if not (0 <= position <= len(self.lattice)):
            raise Exception(f"{position} is outside of the lattice")
        match self.lattice[position]:
            case 1:
                self.tf_position = None
            case 2:
                for rnap_index, rnap_positon in enumerate(self.rnap_positions):
                    if rnap_positon == position:
                        self.rnap_positions.pop[rnap_index]
        self.lattice[position] = 0
        match particle:
            case 1:
                if self.tf_position is not None:
                    raise Exception('There cannot be more than one TF')
                self.tf_position = position
            case 2:
                self.rnap_positions.append(position)
        self.lattice[position] = particle

    def remove_particle(self, position):
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
        # TODO: Implement a way to truncate the lattice so that
        #       only a region of interest is seen
        '''
        Generate a visualization of the lattice
        '''
        if time is None:
            time = self.lattice_age
        plt.axes()
        ax = plt.gca()
        plt.axis('off')
        ax.set_xlim(0, self.lattice_length)
        ax.set_ylim(0, 1.2)
        plt.text(0, 1.1, f"Time: {round(time, 1)}")
        # print(self.lattice)
        # print(self.rnap_positions)
        for site_number in range(self.lattice_length):
            if site_number == self.target_site:
                site_color = 'orange'  # Make target site orange
            else:
                site_color = 'w'
            self.site_drawing(ax, site_number, 0, site_color)
            match self.lattice[site_number]:
                case 1:
                    self.rnap_drawing(ax, site_number, 0)
                case 2:
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
