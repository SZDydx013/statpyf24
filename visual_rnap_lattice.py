#!/usr/bin/env python
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from lattice import Lattice

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
