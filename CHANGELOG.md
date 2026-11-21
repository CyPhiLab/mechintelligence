# Changelog {.unnumbered}

Newest first. Each release is tagged in the repository.

## Unreleased

- The book's subject is now stated as information in the mechanism. Sensor, encoder and
  readout language is gone from the text; Part I's noise is called static noise.
- New research note: how information about the contact propagates through the body. A
  part of the body holds the force through an effective contact damping, and only
  internal dampers between the contact and the part erode it.
- New research note (thread T11): what the body knows about the height of the ground.
  Inside the body height and force propagate alike; at the contact, stiffness helps for
  height and not for force.
- The preface, the first chapter and the open problems now say that contact force is the
  running example and that the environment also includes terrain, ground properties,
  payloads and contact state. New roadmap thread T11.
- New research note: two feet on the ground. The body knows the total load through the
  contact dampers alone; the split between the feet also pays for the dampers on the
  path between them.
- New research note: parts with several coordinates, and branching bodies. A part holds
  the force through the best pattern of forces on it, and in a tree only the path from
  the contact matters at low frequency.
- New research note: how information spreads along a chain. A reciprocity formula gives
  each internal damper's cost, losses on the way to a part accumulate, a static law
  holds on chains and rods, and the SEA's spring deflection holds the force only through
  the rotor's inertia.
- The Langevin chapter compares the whole body with the rotor under thermal and static
  noise. Static energy noise ranks the rotors as thermal noise does; the 5.9-bit
  misranking belongs to a fixed angle noise.

## v0.1 (2026-11-21)

First version of the book.

- Part I, *Information in mechanisms*: the energy metric, its invariance and noise law,
  the port-Hamiltonian structure of the whitened dynamics, bodies with different numbers
  of coordinates (the stiff-SEA limit and the duplicate-readout artifact), attribution and
  morphological computation, and continua, gears and floating bases.
- Part II, *Is energy the right choice?*: the uniqueness theorem, the proof that free
  motion cannot create information under energy noise, the alternatives compared, and the
  Langevin body with its routing results.
- Open problems, the first research note, the roadmap, and the sources appendix.
- 27 tests pin every number in the text.
