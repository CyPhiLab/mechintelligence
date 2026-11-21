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
- **Scope.** The subject is information in the mechanism: what the body holds about the
  world, how it moves through the body, and what dissipation does to it. Sensors,
  readout noise and controllers come later, as their own threads. Results are stated
  for the body and its parts.
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

**T2. How information propagates through the body.** *Active.*

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
- Next step: the matrix version of the reciprocity formula for parts with several
  coordinates, then branching bodies. Fold both notes into @sec-langevin after that.

**T3. The Langevin body as the default noise model.** *Queued.*

- Question: can the discrete-time constant in @prp-contact be proved, not only checked?
- Next step: expand $B_d^\top Q_d^{-1}B_d$ in $\Delta t$, including the position-block
  terms, and show the leading coefficient is $J D^+J^\top/(2\sigma_T^2)$.

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

**T7. Nonlinear and multistable bodies.** *Queued.*

- Question: what does the framework say about mechanical memory, where the noise law is
  the Gibbs law on a non-quadratic potential?
- Next step: a bistable element with an energy barrier; bits stored against barrier
  height and noise energy.

**T8. Latency.** *Queued.*

- Question: when does information about the contact arrive at each part of the body,
  and how does that depend on the transmission? Extends the temporal Shapley analysis.

**T9. Relevance.** *Queued.*

- Question: which measures capture information relevant to the controller or the task,
  such as $I(W;X\mid A)$ or control information?

**T10. Literature.** *Queued.*

- Confirm the *secondary* citations (see T1), then survey morphological computation,
  physical reservoir computing and the thermodynamics of computation.

## Findings log

Newest first. Each entry gives the finding, the test that pins it, and where it is
written up.

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

## Paper candidates

- **The energy metric is unique, and free motion cannot create information under it.**
  @thm-unique and @prp-work, with the comparison of alternatives. Possible venues: a
  robotics venue (RSS, ICRA, IEEE Transactions on Robotics) or a physics or
  information-theory venue.
- **The contact sets the ceiling; internal dissipation decides where information sits.**
  @prp-contact and the propagation results, ideally after threads T2, T3 and T5.
