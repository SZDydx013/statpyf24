#!/usr/bin/env python
'''
Generate a video for a simple scenario
'''
import matplotlib.pyplot as plt
from lattice import Lattice
import matplotlib.animation as animation

visual_lattice = Lattice(
    lattice_length=10,      # We want this to actually be readable
    rnap_attach_rate=0,     # There will be no RNAPs in this (simple case)
    tf_attach_rate=1,
    tf_move_rate=1,
    tf_detach_rate=0,       # TF will never detach
)

(vid_fig, ani) = visual_lattice.visualization_video()
ffwriter = animation.FFMpegWriter(fps=10)
ani.save('results/tf_animation.mp4', writer=ffwriter)
plt.show()
