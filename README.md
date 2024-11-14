# Problem Statement

This model examines how a transcription factor (TF) protein locates a specific binding site on a segment of DNA, even amidst other proteins that move along the same segment. This represents a stochastic (random) process—specifically a first-passage process—where the TF aims to reach its target site in a crowded environment. The process is complicated by other molecules that occupy and move along the same DNA strand, like RNA polymerase (RNAP). This dynamic can be modeled as a one-dimensional lattice where the TF performs a random walk while the RNAP moves unidirectionally along the lattice. The TF and the RNAP molecules cannot occupy the same lattice segment simultaneously in this process. The goal is to use Monte Carlo simulations to estimate the time required for the TF to find its target.

The simplest version of this model reduces the problem to a one-dimensional lattice representing a DNA segment where a single TF particle performs an unbiased random walk. There are no RNAP molecules in this version so the TF is unimpeded as it searches for its binding site. We will simulate the time it takes for the TF to locate its target, which is a designated site on the lattice.

Some parameters we can vary to observe different outcomes include:
- Lattice Size: The length of the DNA segment, impacting the search space and time.
- Attachment Rate: The rate at which the TF attaches to any site on the lattice
- Detachment Rate: The rate at which the TF detaches, resetting its position in solution
- RNAP Density: The density of the RNAPs, which alter the TF’s search dynamics from a traffic-like effect.

Introducing RNAP particles will add complexity to the model, as the RNAP particles will compete with the TF for binding sites. By varying the density of RNAPs and their velocity, we can observe how this “crowded” environment affects the TF’s search time. The RNAPs create a “traffic” effect that either impedes or assists the TF’s movement, depending on the direction of the TF’s flow.
