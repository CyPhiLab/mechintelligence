# Research roadmap {.unnumbered}

This page is the working plan for the book. It lists the open threads and the next step
on each, and keeps a dated record of findings and decisions. It is updated at the end of
every working session. The book itself is not planned in advance; chapters grow out of
the findings recorded here.

## How we work

- **Session protocol.** Each session starts by reading this page and ends by updating it:
  thread statuses, findings, decisions, and a changelog entry if anything was published.
- **From finding to page.** A question becomes a thread here. Exploration happens in
  scratch code. A result that holds up gets a function in `mibook` and a test in
  `tests/`, then a dated research note (status *exploratory*). The note is folded into a
  chapter once its result is checked, its proofs are labelled, and it connects to the
  existing material. The note stays as the dated record and links to the chapter.
- **Checked numbers.** Every number in the text is computed by `mibook` and pinned by a
  test. Figures are code cells that call `mibook.figures`.
- **Proofs.** An incomplete proof is titled "Proof sketch" and states what it assumes.
- **Citations.** Each citation is marked *full text* or *secondary* in the
  [sources appendix](appendices/sources.qmd).
- **Prose.** All text (chapters, notes, this page, the changelog and commit messages)
  is written and edited with the humanizer skill: no staged contrasts, closers or
  inflated claims, no em dashes in prose, plain verbs.
- **Releases.** The live book changes only through a reviewed release. Work in progress
  goes to the `drafts` branch, where every push runs the tests and renders the site
  without deploying. A release is a pull request from `drafts` to `main`; it carries the
  quality-control checklist in `.github/pull_request_template.md` and is merged only
  after the reviewer approves it. Merging deploys the site. Releases get dated tags
  (`v0.1`, `v0.2`, ...) and a changelog entry. Before a result goes to a venue, check that
  venue's preprint policy.
- **Scope.** The subject is information in the mechanism: what the body holds about the
  world, how it moves through the body, and what dissipation does to it. The world is
  more than forces: terrain geometry, ground stiffness and damping, payload mass,
  friction and contact state are environmental quantities too. Contact force is the
  running example, and results should say which quantity they are about. Sensors,
  readout noise and controllers come later, as their own threads. Results are stated
  for the body and its parts.
- **Branches.** `main` is the live book and is changed only by merging a release pull
  request. All other book work goes to `drafts`. Changes to other repositories, such as
  the lab website, go through pull requests.

## Priorities (agreed 2026-11-25)

1. Close the open checks: verify or refute the unchecked links in the
   [landscape note](notes/2026-11-25-landscape.qmd) with `mibook.bench`; confirm the
   *secondary* citations now available in full text (T10, T1). *Done 2026-11-25* for four
   links (two corrected) and for Zahedi and Ay (2013); the others marked secondary in T1
   are not in the collection. The Ljung-Whittle link moves to item 4; the D-optimal
   correspondence needs only a careful statement.
2. First comparison study (T12): every measure, at the book's resolution, on the linear
   benchmarks (legs, network, whisker, swarm). Where do the measures rank designs and parts
   alike, and where do they part? Test the draft axes against the results. *First pass done
   2026-11-25* ([comparison note](notes/2026-11-25-comparison.qmd)); to do: other seeds and
   input levels.
3. Holding, acting and paying on one body (T12): information about the environment,
   empowerment, controller effort and energy cost on the SEA stiffness sweep and the bird's
   shoulder. Framed as an open question, not a trade-off assumed in advance: compliance can
   help action (energy storage and return, impact protection, smoother force, conforming to
   terrain) as well as hurt it, and likewise for perception. Map the regimes where it helps
   or hurts each, and find what decides. *First pass done 2026-11-25*
   ([note](notes/2026-11-25-holding-acting-paying.qmd)): the band relative to the compliant
   element's resonance decides action and withstanding; perception beyond a spring never
   improves; the loss model decides cost. To do: nonlinear effects (stance energy return,
   impacts, conformation).
4. The checks owed under T11 before the parameter notes enter a chapter. *Done
   2026-11-25*, except the question of definition below (see the
   [finite-answers note](notes/2026-11-22-finite-answers.qmd), "Checks of 2026-11-25").
5. Nonlinear comparisons: memory capacity against information on the soft arm; hysteron
   networks at finite temperature (T7, T12).
6. The classification frame in the opening chapter, and the targeted links into
   chapters, once 2 and 3 have tested the axes.
7. Release, on the reviewer's polishing timeline: QC pass and release pull request.

Parked until a study needs them: the fluid-noise question, closed-loop contact through
directed information, T5, T6, compiled force kernels, T4.

## Threads

Order is chosen together at the start of a session. Status is one of *active*, *queued*,
*parked* or *done*.

**T1. Foundations of the energy metric.** *Done*, except for one open item.

- Where: @sec-energy-metric, @sec-right.
- Open: confirm the citations marked *secondary* against the sources. The PDFs that could
  not be fetched automatically are Li, Li and Fritzen (2007); Moore (1981); Kubo (1966);
  Lee, Wensing and Park (2020); Zahedi and Ay (2013); Cover and Thomas, chapters 9 and
  12; Kammer (1991); Geveci (1984), or Baker (1976) or Dupont (1973).

**T2. How information propagates through the body.** *Done.*

- Question: how much of the body's information about the contact does each part of the
  body hold, and what erodes it on the way?
- Done (2026-11-21, [research note](notes/2026-11-21-information-propagation.qmd)): a
  part holds the force through an effective contact damping $b_{\rm eff}(\omega)\ge b_c$,
  and $\rho=b_c/b_{\rm eff}$ is its spectral share. Losses come only from internal dampers
  between the contact and the part. Closed forms for the SEA's rotor and link.
- Done (2026-11-21, [research note](notes/2026-11-21-chains.qmd)): by reciprocity, a
  damper costs its damping times the square of its stretch per unit contact motion when
  the body is driven at the part. In a chain these costs accumulate with distance; the
  static law is $\rho_k(0)=b_c/(b_c+k_c^2\sum_{j<k}b_j/k_j^2)$, mesh-independent on a rod.
  Dampers beyond the part add backflow, and the contact node loses near the modes of the
  body clamped at the contact. The SEA's spring deflection holds the force only through
  the rotor's inertia.
- Done (2026-11-21, [research note](notes/2026-11-21-branching.qmd)): a part with
  several coordinates holds the force through $b_c$ plus the cost of the best force
  pattern on it. A group of chain nodes holds statically what its nearest node holds. On
  the contact side of a bridge element every damper costs the same for all parts beyond.
  In a tree only the path from the contact matters statically; side branches act in band
  through backflow.
- Done (2026-11-21, [research note](notes/2026-11-21-two-contacts.qmd)): with two feet
  down the whole body sees the forces through $B=(JD^{-1}J^\top)^{-1}$ at every frequency.
  For a tree, $B=B_c+b_\times L$ with $b_\times$ the dampers between the feet in series:
  the total load is seen through the contact dampers alone, the split also pays for the
  path between the feet. A one-coordinate part sees one mix of the forces, weighted by how
  much each foot moves when the part is pushed.
- Done (2026-11-21): the four notes are folded into @sec-inside (Part III), together with
  the terrain-height result of T11. The notes stay as dated records.

**T3. The Langevin body as the default noise model.** *Done.*

- Question: can the discrete-time constant in @prp-contact be proved, not only checked?
- Done (2026-11-25): proved through relative entropy. Sampling never beats the
  continuous-time (Girsanov) rate, $g\le\Delta t\,JD^+J^\top/(2\sigma_T^2)$ at every step,
  and one explicit linear statistic reaches it to first order. Both steps are the
  Cauchy–Schwarz inequality; @rem-kl reads them as data processing between path and sample.
  Checked by `tests/test_bridges.py`, including singular $D$ and a case outside the
  hypothesis.
- Open: the gap below the bound closes like $\Delta t^2$ in every case checked; only
  $O(\Delta t)$ is proved. The ground-stiffness rate is not covered, because there the
  parameter also changes the noise covariance.

**T4. Predictive validation on the Go2 designs.** *Parked: needs data.*

- Question: which of the total information, the share held by a part and $I(W;X\mid A)$
  best predicts
  closed-loop tracking error across designs and seeds?
- Depends on: the RL training pipeline and trained policies from the quadruped
  experiments, which are not in either repository.

**T5. The Langevin body in the MuJoCo leg.** *Queued.*

- Question: does the analytic Langevin result (flat whole-body bits, rotor bits climbing
  with stiffness) hold in the nonlinear simulated leg?
- Next step: inject damping-matched noise forces in the simulation and estimate with
  InfoNCE. Needs the `sim` extra (torch and warp), ideally on a GPU.

**T6. Floating bases.** *Queued.*

- Question: how should base positions that store no energy be treated?
- Options: keep velocities only; regularize with $\omega_{\rm ref}^2H$ and sweep
  $\omega_{\rm ref}$; use gravity and contact stiffness at the operating point.

**T7. Nonlinear bodies.** *Active.*

- Question: what carries over when the body is nonlinear? Everything so far is linear,
  and the energy metric is a fixed quadratic form.
- Kinds to treat: nonlinear springs (stiffening, softening), configuration-dependent
  inertia $H(q)$ at large motions, contact that switches on and off (lift-off, impacts,
  friction), and multistable elements that store bits (mechanical memory).
- What changes in the framework: the Gibbs law becomes $\exp(-V/\sigma_T^2)$ on a
  non-quadratic potential, so noise is no longer Gaussian; the energy metric becomes
  local, along a trajectory (@rem-scope); and information has to be estimated without
  the Gaussian formulas, for example by simulation with a neural estimator (as in T5).
- Done (2026-11-22, [research note](notes/2026-11-22-mechanical-bit.qmd)): a bistable
  element keeps a bit written by the environment for a time growing like
  $e^{\Delta E/\sigma_T^2}$ (Kramers; simulated rates 7% to 15% below the formula), while a
  harmonic element forgets the same push in a few relaxation times. For a Markov body, a
  window of history holds exactly what its first state holds about a past event.
- Done (2026-11-22, [research note](notes/2026-11-22-writing-a-bit.qmd)): writing the
  bit. A sharp kick bounces back and records at most 0.72 bits (barrier $8\sigma_T^2$); a
  steady push beyond the critical force $8\Delta E/(3\sqrt3a)$ records 0.93 to 0.96 bits at
  3.6 to 4.3 barriers of work, so the cost grows with the log of the bit's lifetime.
- Done (2026-11-22, [research note](notes/2026-11-22-bit-in-a-body.qmd)): a bit inside
  a body. The bit shifts the body's static shape; a part reads it only if the element sits
  in its load path (on the SEA, a bistable series element is invisible to the link); the
  body's push-back $k_{\rm eff}$ makes the bit readable and lowers its barrier by
  $(1-2k_{\rm eff}/k_w)^2$; reading a small shift takes a window, at the zero-frequency
  push-on-the-part rate.
- Done (2026-11-22, [research note](notes/2026-11-22-erasing-a-bit.qmd)): erasure. The
  four-stage protocol costs $1.37\,\sigma_T^2$ at $\tau=5$ and $0.695\,\sigma_T^2$ at
  $\tau=200$, approaching Landauer's $\sigma_T^2\ln2$ with a $1/\tau$ excess; slow erasure
  leaks (1.2% errors). In a cycle the push-driven write ($\approx29\,\sigma_T^2$) dominates;
  reading costs no work.
- Done (2026-11-25): the four T7 notes are folded into @sec-memory (Part IV). The notes
  stay as dated records.
- Next steps: a part watching a bit being written; erasing a bit inside a body; a
  stiffening ground spring.

**T8. Latency.** *Queued.*

- Question: when does information about the contact arrive at each part of the body,
  and how does that depend on the transmission? Extends the temporal Shapley analysis.

**T9. Relevance.** *Queued.*

- Question: which measures capture information relevant to the controller or the task,
  such as $I(W;X\mid A)$ or control information?

**T11. Environments beyond force.** *Active.*

- Question: what does the body hold about environmental quantities other than a contact
  force, and do the T2 results carry over?
- Three kinds to treat. Prescribed motions, such as terrain height $z$, which enter
  through the contact spring and damper as the force $(k_c+i\omega b_c)z$. Parameters of
  the dynamics, such as ground stiffness, ground damping, payload mass and friction,
  which change $K$, $D$ or $H$ and the energy metric with them. Discrete states, such as
  contact or no contact and contact location, which change $J$.
- Done (2026-11-21, [research note](notes/2026-11-21-terrain.qmd)): terrain height
  enters as the force $k_cz+b_c\dot z$. Inside the body nothing changes (each part holds
  the same spectral share as for force); at the contact the two questions part ways.
  Stiffer ground helps for height and not for force; damping always hurts for force and
  hurts for height only up to $b_c=k_c/\omega$.
- Folded into @sec-terrain.
- Done (2026-11-22, [research note](notes/2026-11-22-ground-stiffness.qmd)): ground
  stiffness as an unknown parameter. One thermal snapshot of a part carries Fisher
  information $\frac12\varepsilon^2$ about $\log k_c$, with $\varepsilon$ the ground
  spring's share of the stored energy when the part is pushed. The whole body learns
  $\log k_c$ at $k_c/(2b_c)$ per second for every design (proof sketch, confirmed
  numerically). The SEA's rotor learns far less about stiffness than about force.
- Done (2026-11-22, [research note](notes/2026-11-22-mass-and-resolution.qmd)): payload
  mass. A velocity snapshot follows the same law with kinetic energy. Over time,
  information about masses and dampings grows with the band of motion resolved,
  $(2\varepsilon)^2$ per hertz for the link mass and 1 per hertz for the ground damping;
  stiffness settles. A thermal bath with finite memory (Maxwell elements, consistent with
  fluctuation-dissipation, `mibook.bath`) gives the same growth, so the cause is inertia,
  not white noise.
- Done (2026-11-22, [research note](notes/2026-11-22-finite-answers.qmd)): which
  properties have a finite answer without an outside scale. A part's high-frequency
  spectrum is $c/\omega^4$ with $c=2\sigma_T^2p^\top H^{-1}DH^{-1}p$; the information rate
  about a property is finite exactly when it leaves $c$ unchanged, and otherwise grows by
  $(\partial_\theta\log c)^2$ per hertz. A part learns without limit only about what it
  touches: its own mass and the dampers attached to it.
- Decided (2026-11-22): answers about the masses and dampers a part touches, and every
  mass or damping question about the whole body, are given for a stated band, 100 Hz by
  default.
- To settle before these notes go into a chapter, since the reviewer is not yet
  convinced: confirm Whittle (1953) and Ibragimov and Rozanov (1978) against the sources;
  check the finite-window version directly (exact Gaussian Fisher information over longer
  windows, not only Whittle's formula); check the several-coordinate form and a chain;
  and ask whether a part's own mass and dampers belong to "the environment" at all.
- Done (2026-11-25): the Whittle-type formula confirmed in Ljung (1999), Eqs. 13.24–13.26
  (the 1953 and 1978 originals remain unread); exact finite-window Fisher information in
  linear time agrees with the spectral rates to four digits and reproduces the growth of
  4.00 per hertz for the link's mass; the several-coordinate form,
  $\operatorname{tr}[(c^{-1}\partial c)^2]$ per hertz, agrees with exact windows on a chain to
  three decimals. A caution: spectral rates of lightly damped bodies need very fine
  frequency grids; the exact window method has no grid.
- Open, for the reviewer: is a part's own mass (or a damper attached to it) "environment"?
  Proposed answer: a property is environment when it can change from task to task without
  redesigning the body (a payload, the ground); the body's own masses are design. On that
  reading, the only properties that need a stated band are environmental masses and dampers
  attached directly to the part reading them, such as a payload on the link read at the link.
- Next step after the decision: payload on the SEA and on the bodies of @sec-inside;
  whether $\varepsilon$ predicts what parts of chains and trees hold about the ground.

**T10. Literature.** *Active.*

- Done (2026-11-25): first survey of 95 papers supplied by the reviewer, read in nine
  topic groups. One card per paper and a first map of the field are kept with the
  literature collection. The [landscape note](notes/2026-11-25-landscape.qmd) records five
  meanings of mechanical intelligence, draft axes, four proved links, and links still to
  check.
- Open: confirm the *secondary* citations (see T1); several sources, such as Zahedi and Ay
  (2013), are now in the collection in full text. Check the links listed as unchecked in the
  landscape note.

**T12. Common benchmarks for different approaches.** *Active.*

- Question: computed at the book's resolution on the same bodies, where do the measures of
  other approaches agree with the book's, and with each other, and where do they part?
  Measures: information held (the book), memory capacity, Zahedi and Ay's measures,
  transfer entropy, empowerment, control effort, and energy cost.
- Principle: every benchmark is *ready to go*, not fully studied. Ready means a model, a
  simulator with a declared noise, a connection to the shared measures, a test, and one
  baseline number. Studies are run when a question needs them.
- Code (`mibook.bench`), in three layers, so each body and each measure is written once:
  1. *Bodies.* One small interface: state, a stochastic step with declared noise, named
     environment channels (inputs such as forces, gusts and terrain; properties such as
     stiffness, mass and flow speed), and a linearization at an operating point where one
     exists.
  2. *Engines.* An exact engine for linear bodies with Gaussian noise, using the filtering
     (innovations) form so the cost grows linearly with the window, and frequency-domain
     formulas for rates. A sampling engine that simulates batches of trajectories for
     nonlinear and contact bodies, checked against the exact engine on linear ones.
  3. *Measures.* Information held about an input or a property, Fisher information, memory
     capacity, Zahedi and Ay's measures, transfer entropy, empowerment, energy cost and
     entropy production; control effort later.
- Benchmarks, in tiers:
  - *Linear, closed forms:* the rigid and SEA legs (ported first; must reproduce the
    book's numbers), random mass-spring-damper networks (the standard body of
    morphological and reservoir computing), a whisker-like beam touched along its length.
  - *Nonlinear, smooth:* a soft arm with stiffening springs; a chain of bistable elements
    (hysterons at finite temperature).
  - *Contact and switching:* a spring-loaded hopper; a compliant gripper holding objects of
    unknown mass or stiffness; a passive walker on an unknown slope; a soft tentacle
    grasping (planar discretized rod with contact); a hyper-redundant many-legged robot on
    rough terrain.
  - *Fluids:* a foil in a flow, including a vortex street (reduced-order unsteady
    thin-airfoil model with the street as a travelling disturbance; a point-vortex model as
    an upgrade; no CFD); gust rejection in a bird (longitudinal flight dynamics with a
    compliant wing root, a gust-response filter and a turbulence spectrum).
  - *Collectives:* a swarm of simple robots coupled physically or by local sensing.
- Open modeling question for the fluid benchmarks: turbulence is naturally part of the
  environment, so the resolution must come from the body's own dissipation. Settle it when
  those benchmarks are first used.
- Order: interface and engines; legs (regression against the book); linear tier;
  nonlinear tier; contact, fluid and collective tiers. Several sessions; the foil and the
  bird need the most care.
- Then: compute the measures on chosen benchmarks and test the draft axes against the
  results.

## Findings log

Newest first. Each entry gives the finding, the test that pins it, and where it is
written up.

- **2026-11-25.** The T11 checks: exact finite-window Fisher information (linear time, no
  Whittle formula) matches the spectral rates on the leg to four digits; the link's mass
  grows by 4.00 per hertz, the rotor's stays at 31.1 per second; the several-coordinate
  form $\operatorname{tr}[(c^{-1}\partial c)^2]$ matches a chain to three decimals; Ljung (1999)
  confirms the Whittle-type formula. `tests/test_bench_t11.py`.
- **2026-11-25.** Compliance mapped across stiffness and frequency on the leg and the bird.
  A series spring raises ground force per torque from about $f_r/3$ to $1.1f_r$ (up to
  5.1 times) and lowers it above; it amplifies shocks at the rotor in that band and isolates
  above $1.25f_r$. The rotor never perceives the ground better than the rigid leg's rotor,
  and the spring never saves mechanical work per unit ground force (both proved); motor heat
  per unit force falls in the resonance band. The bird's shoulder perceives gusts equally at
  every stiffness (proved). `tests/test_bench_regimes.py`;
  [note](notes/2026-11-25-holding-acting-paying.qmd).
- **2026-11-25.** Measures compared on four families of linear benchmarks: measures about
  the contact force rank leg designs identically while $\mathrm{MC}_W$ reverses them; memory
  capacity ranks network nodes unlike window information (0.35); the nodes that hold the
  input hold least about the anchors ($-0.79$); the motor's empowerment over the foot peaks at
  intermediate series stiffness; every swarm member acts alike on the swarm's mean while
  perceiving differently; and the input's band can reverse which member knows most (a new
  axis). `tests/test_bench_compare.py`; [comparison note](notes/2026-11-25-comparison.qmd).
- **2026-11-25.** Four links from the literature checked: with the whole state observed, the
  book's information about a white force is accumulated one-step transfer entropy and equals
  directed information; for a part, one-step transfer entropy misses nearly everything
  ($9\times10^{-7}$ against 0.093 bits per sample on the SEA rotor). Zahedi and Ay's measures
  coincide for a passive body and separate under random actions; feedback raised
  $\mathrm{MC}_W$ on the SEA, against the reading notes' guess. Empowerment bounds the
  information at equal power (5.10 against 1.69 bits). A part's effective contact damping is
  the dissipation, hence the entropy production, per squared contact speed when the part is
  shaken. `tests/test_bridges.py`; [landscape note](notes/2026-11-25-landscape.qmd).
- **2026-11-25.** The constant in @prp-contact is proved: $g\le\Delta t\,JD^+J^\top/(2\sigma_T^2)$
  at every sampling step, with equality to first order as $\Delta t\to0$. The information
  held about a force equals the KL control cost of applying it (@rem-kl). Four further links
  between the book and other approaches are proved: memory capacity and information share
  one Gramian spectrum; the thermal-force Gramian is the Gibbs law; $\mathrm{MC}_W$ is the
  excess entropy of the thermal motion; and the energy-share rule for a spring in series
  with the ground. `tests/test_bridges.py`;
  [landscape note](notes/2026-11-25-landscape.qmd).

- **2026-11-22.** Erasing a mechanical bit costs at least $\sigma_T^2(\ln2-H(p))$ for error
  $p$; the simulated four-stage erasure costs 1.372 down to 0.695 $\sigma_T^2$ as each stage
  lengthens from 5 to 200 time units, with a $1/\tau$ excess ($B=4.2$), and stays above
  its bound at every speed (`tests/test_erasure.py`;
  [research note](notes/2026-11-22-erasing-a-bit.qmd)).

- **2026-11-22.** A bit inside a body: a part reads it only through a load path. A
  bistable series element on the SEA moves the rotor by its full stroke and the link by
  exactly zero; a bistable motor latch is read by the link at 0.38 bits from a snapshot
  and 0.98 bits from 10 ms, and needs a barrier of 12 thermal energies to keep 8 inside the
  body. A nonlinear simulation matches the link's predicted shift to 3% and spread to 1%
  (`tests/test_bitbody.py`; [research note](notes/2026-11-22-bit-in-a-body.qmd)).

- **2026-11-22.** Writing a mechanical bit: a steady push at 1.1 to 1.3 times the critical
  force writes 98.5% to 99.4% of the time (0.93 to 0.96 bits) for 3.6 to 4.3 barriers of
  work; the best sharp kick records 0.72 bits because the element bounces back. At a
  barrier of $4\sigma_T^2$, 8% flip on their own and no write beats 0.57 bits
  (`tests/test_bistable.py`; [research note](notes/2026-11-22-writing-a-bit.qmd)).

- **2026-11-22.** A mechanical bit: from a barrier of 2 to 10 $\sigma_T^2$ the half-life of
  a bit in a bistable element grows 1,100-fold, while the same push in a harmonic element
  is forgotten after 1.3 to 2.2 time units. A window of a Markov body's history holds
  exactly what its first state holds about a past event (checked to $10^{-9}$ on the rigid
  leg) (`tests/test_bistable.py`; [research note](notes/2026-11-22-mechanical-bit.qmd)).

- **2026-11-22.** A part's history holds a finite amount per second about a property
  exactly when the property leaves the part's leading high-frequency coefficient
  $p^\top H^{-1}DH^{-1}p$ unchanged; otherwise the information grows by
  $(\partial_\theta\log c)^2$ per hertz. On the SEA all ten property-part pairs match: the
  rotor holds a finite 31.2 per second about the link mass while the link gains 4 per
  hertz (`tests/test_scale.py`; [research note](notes/2026-11-22-finite-answers.qmd)).

- **2026-11-22.** Information about masses and dampings grows with the band of motion
  resolved, because inertia sets the motion above a body's modes: on the rigid leg the
  link mass gains $(2\varepsilon)^2=2.25$ per second per hertz ($\varepsilon=0.75$) and the
  ground damping 1 per hertz, with white noise or with a bath of 0.1 or 1 ms memory. The
  0 to 100 Hz band holds 205 per second about the mass. Ground stiffness settles at 833
  per second. A velocity snapshot carries $\frac12\varepsilon^2$ about the mass
  (`tests/test_parameters.py`, `tests/test_bath.py`;
  [research note](notes/2026-11-22-mass-and-resolution.qmd)).
- **2026-11-22.** Correction before release: the first draft of the mass note blamed white
  thermal noise and proposed a bath with finite memory as the fix. Building that bath
  showed the growth is unchanged, and the note now gives the inertia explanation.

- **2026-11-22.** The body learns the ground's stiffness from its own thermal motion. A
  snapshot of a part carries Fisher information $\frac12\varepsilon^2$ about $\log k_c$
  ($\varepsilon$: the ground spring's energy share under a push at the part); the whole
  body learns at $k_c/(2b_c)=833$ per second in every design (71.2 in 85 ms, the
  stiffness to about 12%). The SEA's rotor holds 0.02% of that at $k_s=70$, against 6% of
  the force information (`tests/test_parameters.py`;
  [research note](notes/2026-11-22-ground-stiffness.qmd)).

- **2026-11-21.** Terrain height reaches the body as the force $k_cz+b_c\dot z$. Every
  design holds 38.21 bits about it, and each part's share matches its share of force
  information. On the rigid leg, height bits rise from 13.7 to 71.2 as the ground
  stiffens 16-fold while force bits stay at 37.74; ground damping lowers force bits
  monotonically but height bits only down to a minimum near $b_c=k_c/\omega$
  (`tests/test_terrain.py`; [research note](notes/2026-11-21-terrain.qmd)).

- **2026-11-21.** Two feet on the ground: the body always knows the total load through
  the contact dampers alone (47.22 bits in the stance example, whatever the internal
  damping), while the split between the feet pays for the dampers on the path between
  them (47.06 down to 37.24 bits as that damping grows 100-fold). Each foot holds its own
  force; the trunk holds the total load (8.45 bits) much better than the split (3.34)
  (`tests/test_contacts.py`; [research note](notes/2026-11-21-two-contacts.qmd)).

- **2026-11-21.** A part with several coordinates holds the force through the best
  pattern of forces on it. Statically a group of chain nodes holds what its nearest node
  holds, and in a tree only the path to the contact counts; in band a far node adds (node
  1 of the uniform chain holds 0.381, nodes 1 and 7 together 0.499) and side branches
  feed back (heavier swing-leg damping lowers the trunk from 0.242 to 0.214)
  (`tests/test_chain.py`; [research note](notes/2026-11-21-branching.qmd)).

- **2026-11-21.** Reciprocity gives the cost of every internal damper: drive the body at
  the part, and a damper costs $b_j|\text{stretch}_j/\text{contact motion}|^2$. Along a
  chain the costs of dampers on the way accumulate, and statically each costs
  $b_j(k_c/k_j)^2$. On a free rod the share converges under mesh refinement, and the
  contact point holds the least in band, losing near the modes of the rod clamped at the
  contact. The SEA's spring deflection holds 57% to 36% of the body's information,
  falling with stiffness (`tests/test_chain.py`;
  [research note](notes/2026-11-21-chains.qmd)).

- **2026-11-21.** Information is lost on its way through the body only at internal
  dampers. A part holds the contact force through
  $b_{\rm eff}(\omega)=b_c+1/(2\sigma_T^2q)$; with no internal damper every part holds
  everything. On the SEA at $k_s=70$ the link holds 98% of the body's information and the
  rotor 6%, and lowering the series damping raises the rotor's share
  (`tests/test_propagation.py`;
  [research note](notes/2026-11-21-information-propagation.qmd)).
- **2026-11-21.** Energy is the unique noise metric under five requirements, and
  invariance under free motion is equivalent to "free motion never creates
  information" (`test_uniqueness_step_four_identity`,
  `test_free_motion_creates_information_only_without_energy_noise`; @sec-right).
- **2026-11-21.** The frequency-weighted family $W_p$ converges in the stiff limit
  exactly when $p<2$; only action weighting ($p=-1$) also passes mesh refinement
  (`test_frequency_weighted_stiff_limit_threshold_is_p_equals_two`,
  `test_frequency_weighted_mesh_behaviour`; @sec-alternatives).
- **2026-11-21.** Data-driven whitening gives every body the same bits
  (`test_data_driven_whitening_erases_the_body`; @prp-datadriven).
- **2026-11-21.** Langevin body: the whole body holds the same bits in every design, set
  by the contact damping $JD^+J^\top=1/b_i$. The rigid leg's rotor holds 99.9% of that;
  the SEA's rotor climbs from 6% to 99.5% with stiffness. Static energy noise ranks the
  rotors the same way, while a fixed angle noise lets a resonant SEA win by up to 5.9 bits
  (`test_langevin_full_state_information_depends_only_on_contact_damping`,
  `test_langevin_information_reaching_the_rotor`; @sec-langevin).
- **2026-11-20.** Coordinate invariance, the port-Hamiltonian structure, the stiff-SEA
  limit with explicit rates, the 8.4716-bit double-counting artifact, SNR asymptotics,
  consistent Shapley noise, the closed form for $\mathrm{MC}_W$, mesh independence and
  the floating-base condition (the remaining tests in `tests/test_energy_whitening.py`;
  Part I).

## Decisions log

- **2026-11-21.** The work becomes a live textbook on GitHub Pages, published as results
  are checked; important results go to venues later. The book is not planned in advance.
- **2026-11-21.** Quarto for authoring (one source for the website and the PDF; numbered
  theorems with cross-references; figures from code).
- **2026-11-21.** A dedicated repository, `CyPhiLab/mechintelligence`, owned by the lab.
  It depends on the analysis library in `zpatty/quadruped-mech-info`, pinned to commit
  `2b25bca`, so the book's numbers stay reproducible.
- **2026-11-21.** The book is served at `mechintelligence.cyphilab.com` and linked from
  the lab website's navigation bar (CyPhiLab/CyPhiLab.github.io pull request 3).
- **2026-11-21.** Two-tier publishing: dated research notes, and chapters that absorb them
  once stable.
- **2026-11-21.** All prose goes through the humanizer skill.
- **2026-11-21.** Branch workflow: direct pushes to `main` for book content; pull
  requests for other repositories.
- **2026-11-21.** The book is live at `mechintelligence.cyphilab.com`, tagged `v0.1`. The
  Publish workflow installs Quarto and TinyTeX with `quarto-dev/quarto-actions/setup`,
  because `quarto install tinytex` hit GitHub's anonymous API rate limit on the shared
  runners.

- **2026-11-21.** Scope rule: the book is about information in the mechanism. Sensors,
  encoders and readout noise are out of scope until the mechanism is understood, and
  return later as their own threads. Thread T2 was reframed from a sensor's routing
  efficiency to the share of information held by a part of the body. The Langevin
  chapter now compares the whole body and the rotor, and Part I's noise is called static
  noise.

- **2026-11-21.** Environments are more than forces. Contact force stays the running
  example, but terrain geometry, ground parameters, payload and contact state are in
  scope, and results name the quantity they are about. New thread T11.

- **2026-11-21.** Thread T2 and the first T11 result are folded into a new Part III,
  "Information inside the body" (@sec-inside), rather than into the Langevin chapter:
  the material is about where information sits, not about whether energy is the right
  metric. Promoted notes become short dated records so theorem labels stay unique.

- **2026-11-22.** The live book is no longer updated as results come in. Work goes to
  the `drafts` branch, which tests and renders but does not deploy; the live book changes
  only when the reviewer approves a release pull request from `drafts` to `main`, with a
  quality-control checklist.
- **2026-11-22.** Nonlinear bodies are in scope for later: thread T7 widened from
  multistable elements to nonlinear springs, configuration-dependent inertia, switching
  contact and mechanical memory.

- **2026-11-22.** The book keeps the history view: what a part holds is what its whole
  trajectory holds, because computation over time will need it. Defining "holding" by the
  instantaneous state was considered and set aside.

- **2026-11-22.** Masses and dampings are reported for a stated band, 100 Hz by default,
  where the criterion of the finite-answers note says a band is needed. Accepted with
  reservations; the checks listed under T11 come first.

- **2026-11-25.** The live site is taken offline until the reviewer has polished the book;
  work continues on `drafts`, and the next release republishes it. The nonlinear notes
  are folded into a new Part IV, "Nonlinear bodies".

- **2026-11-25.** The book's goal is a map of mechanical intelligence: a way to classify
  it, and the relationships between the different ways people have thought about it. Other
  approaches are treated as other views of the same subject, not as rivals, and they are
  woven into the chapters where they bear on the question rather than kept in a separate
  related-work section. Where two approaches overlap, the book should say how they relate:
  when one reduces to the other, what each assumes, and what each can see that the other
  cannot. The literature informs the map but does not limit it; new ideas are welcome
  where the map has gaps.

## Paper candidates

- **The energy metric is unique, and free motion cannot create information under it.**
  @thm-unique and @prp-work, with the comparison of alternatives. Possible venues: a
  robotics venue (RSS, ICRA, IEEE Transactions on Robotics) or a physics or
  information-theory venue.
- **The contact sets the ceiling; internal dissipation decides where information sits.**
  @prp-contact and @sec-inside (reciprocity, path laws, two contacts, terrain), ideally
  after threads T3 and T5.
