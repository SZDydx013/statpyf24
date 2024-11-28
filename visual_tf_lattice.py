#!/usr/bin/env python
"""
This version assumes the following:

    TF attaches at any site along the lattice with equal probability
    TF has no rate of detachment and will walk unitl it reaches the target site
    There are no RNAPs present to impede the path

"""
import matplotlib.pyplot as plt
from lattice import Lattice

sample_lattice = Lattice(
    lattice_length=10,      # We want this to actually be readable
    rnap_attach_rate=0,     # There will be no RNAPs in this
    tf_attach_rate=1,       # TF will attach on the first step
    tf_move_rate=1,         # TF will move on every step
    tf_detach_rate=0,       # TF will never detach
)

ani = sample_lattice.visualization_video()

plt.show()
