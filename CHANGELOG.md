# Changelog {.unnumbered}

Newest first. Each release is tagged in the repository.

## Unreleased

- The soft arm is now a Cosserat rod simulated with PyElastica (optional `physics` extra,
  with MuJoCo), anchored to the silicone arm of Nakajima et al.; it absorbs the tentacle
  benchmark and can grasp a cylinder. It passes textbook checks (static tip deflection,
  first bending frequency) and a grasp check. The earlier spring-ladder arm is kept as the
  test body `ladder_arm`. The roadmap now records the benchmark standard and audit.
- General handling of unstable bodies in `mibook.bench` (priors, steady-state rates,
  canonical mirror stabilization, nonlinear mirror feedback), a `pendulum` benchmark, and a
  research note. `mibook.bench.animate` renders video demos of every benchmark.
- New research note (T7, T12): transforming and storing in nonlinear bodies, with a figure
  (`fig_nonlinear`) and code `mibook.bench.nonlinear_study`.
- The checks owed for the parameter notes (T11): exact finite-window Fisher information in
  linear time (`linear.window_fisher`, `mibook.bench.t11`), agreement with the spectral
  rates, the several-coordinate form of the leading-coefficient theorem, and the formula
  confirmed in Ljung (1999). Written up in the finite-answers note.
- New research note (T12): where compliance helps and hurts action, withstanding,
  perception and cost, on the leg and the bird, with a figure (`fig_regimes`) and code
  `mibook.bench.regimes`.
- New research note (T12): the measures of different approaches compared on the same linear
  bodies, with a figure (`fig_compare`) and the comparison code `mibook.bench.compare`. The
  legs gain a motor channel and a foot part.
- The landscape note gains four checked links (transfer entropy and directed information,
  Zahedi and Ay's measures in closed form, empowerment as a bound, and effective damping as
  dissipation per squared contact speed), with two corrections to the literature's guesses.
- Zahedi and Ay (2013) confirmed against the full text; the attribution chapter now states
  their normalization.
- New benchmark library `mibook.bench` (thread T12): fourteen benchmark bodies across
  linear, nonlinear, contact, fluid and collective tiers, an exact linear engine that runs
  in time linear in the window, a sampling engine for nonlinear bodies, and the shared
  measures. Code only; nothing in the book's text uses it yet.
- @prp-contact is now proved, not only checked (thread T3), with a remark (@rem-kl) that
  reads the bound as relative entropy and links it to KL control.
- New research note (threads T10 and T12): how other approaches to mechanical intelligence
  relate to the book, from a first survey of 95 papers, with four proved links checked by
  `tests/test_bridges.py`.
- New Part IV, *Nonlinear bodies*, with one chapter (@sec-memory) that folds in the
  research notes on a mechanical bit, writing it, a bit inside a body, and erasure. The
  notes stay as short dated records.
- New research note (thread T7): erasing a mechanical bit, and the cost of a
  write-read-erase cycle against Landauer's bound.
- New research note (thread T7): a mechanical bit inside a body. A part reads a stored
  bit only if the element is in its load path, and the push-back that makes it readable
  also lowers its barrier.
- New research note (thread T7): writing a mechanical bit. A steady push beyond a
  critical force writes reliably; a sharp kick bounces back. The bit note now says what
  the wells are physically.
- New research note (thread T7, nonlinear bodies): a mechanical bit. A barrier keeps a
  record of an event exponentially longer than a linear body can, and a window of history
  holds what its first state holds about the past.
- New research note (thread T11): what a part's history can say without limit. One
  coefficient of the part's high-frequency motion decides whether its information about
  a property of the environment is finite; a part learns without limit only about its
  own mass and the dampers attached to it.
- New research note (thread T11): payload mass, and why some properties need a stated
  band. Information about masses and dampings grows with the band of motion resolved,
  because inertia sets the motion above a body's modes; a thermal bath with finite memory
  (new module `mibook.bath`) does not change this. A new open-problems limitation records
  it.
- New research note (thread T11): what the body knows about the stiffness of the ground.
  Its thermal motion carries it; the whole body learns it at half the ground's corner
  frequency in every design, and a part in proportion to the ground spring's share of
  its stored energy.
- The live book now changes only through reviewed releases: work goes to a `drafts`
  branch that tests and renders without deploying, and a release is a pull request to
  `main` with a quality-control checklist. Roadmap thread T7 now covers nonlinear bodies
  in general.
- The book's subject is now stated as information in the mechanism. Sensor, encoder and
  readout language is gone from the text; Part I's noise is called static noise.
- New research note: how information about the contact propagates through the body. A
  part of the body holds the force through an effective contact damping, and only
  internal dampers between the contact and the part erode it.
- New Part III, *Information inside the body*, with one chapter (@sec-inside) that folds in
  the research notes on propagation, chains, branching, two feet and terrain height. The
  notes stay as short dated records.
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
