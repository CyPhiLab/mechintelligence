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
- **Releases.** Dated tags (`v0.1`, `v0.2`, ...) with a changelog entry. Before a result
  goes to a venue, check that venue's preprint policy.
- **Branches.** Book content goes straight to `main`, which deploys the site. Changes to
  other repositories, such as the lab website, go through pull requests. This is a
  default and can change.

## Threads

Order is chosen together at the start of a session. Status is one of *active*, *queued*,
*parked* or *done*.

**T1. Foundations of the energy metric.** *Done*, except for one open item.

- Where: @sec-energy-metric, @sec-right.
- Open: confirm the citations marked *secondary* against the sources. The PDFs that could
  not be fetched automatically are Li, Li and Fritzen (2007); Moore (1981); Kubo (1966);
  Lee, Wensing and Park (2020); Zahedi and Ay (2013); Cover and Thomas, chapters 9 and
  12; Kammer (1991); Geveci (1984), or Baker (1976) or Dupont (1973).

**T2. Retained versus routed information.** *Queued.*

- Question: what is the right definition of routing efficiency (sensor bits divided by
  full-state bits under Langevin noise), and what are its properties? Candidates to prove:
  bounds, invariance under relabeling the sensor, behaviour under composition of
  mechanisms.
- Next step: state the definition, prove the bound $0\le\eta\le1$ and the invariances,
  and compute $\eta$ for the joint encoder as well as the motor encoder.

**T3. The Langevin body as the default sensor model.** *Queued.*

- Question: can the discrete-time constant in @prp-contact be proved, not only checked?
- Next step: expand $B_d^\top Q_d^{-1}B_d$ in $\Delta t$, including the position-block
  terms, and show the leading coefficient is $J D^+J^\top/(2\sigma_T^2)$.

**T4. Predictive validation on the Go2 designs.** *Parked: needs data.*

- Question: which of retention, routing efficiency and $I(W;X\mid A)$ best predicts
  closed-loop tracking error across designs and seeds?
- Depends on: the RL training pipeline and trained policies from the quadruped
  experiments, which are not in either repository.

**T5. The Langevin body in the MuJoCo leg.** *Queued.*

- Question: does the analytic Langevin result (flat full-state bits, motor bits climbing
  with stiffness) hold in the nonlinear simulated leg?
- Next step: inject damping-matched noise forces in the simulation and estimate with
  InfoNCE. Needs the `sim` extra (torch and warp), ideally on a GPU.

**T6. Floating bases.** *Queued.*

- Question: how should base positions that store no energy be treated?
- Options: keep velocities only; regularize with $\omega_{\rm ref}^2H$ and sweep
  $\omega_{\rm ref}$; use gravity and contact stiffness at the operating point.

**T7. Nonlinear and multistable bodies.** *Queued.*

- Question: what does the framework say about mechanical memory, where the noise law is
  the Gibbs law on a non-quadratic potential?
- Next step: a bistable element with an energy barrier; bits stored against barrier
  height and noise energy.

**T8. Latency.** *Queued.*

- Question: when does routed information arrive at the sensor, and how does that depend
  on the transmission? Extends the temporal Shapley analysis.

**T9. Relevance.** *Queued.*

- Question: which measures capture information relevant to the controller or the task,
  such as $I(W;X\mid A)$ or control information?

**T10. Literature.** *Queued.*

- Confirm the *secondary* citations (see T1), then survey morphological computation,
  physical reservoir computing and the thermodynamics of computation.

## Findings log

Newest first. Each entry gives the finding, the test that pins it, and where it is
written up.

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
- **2026-11-21.** Langevin body: with the full state observed, every design gives the
  same bits, set by the contact damping $JD^+J^\top=1/b_i$. The rigid transmission routes
  99.9% of that to its motor encoder; the SEA climbs from 6% to 99.5% with stiffness.
  Readout-only noise misranks resonant SEAs by up to 5.9 bits
  (`test_langevin_full_state_information_depends_only_on_contact_damping`,
  `test_langevin_routing_reproduces_the_proprioceptive_argument`; @sec-langevin).
- **2026-11-20.** Coordinate invariance, the port-Hamiltonian structure, the stiff-SEA
  limit with explicit rates, the 8.4716-bit duplicate-readout artifact, SNR asymptotics,
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

## Paper candidates

- **The energy metric is unique, and free motion cannot create information under it.**
  @thm-unique and @prp-work, with the comparison of alternatives. Possible venues: a
  robotics venue (RSS, ICRA, IEEE Transactions on Robotics) or a physics or
  information-theory venue.
- **Morphology routes information; the contact sets the ceiling.** @prp-contact and the
  routing results, ideally after threads T2, T3 and T5.
