# `mibook.bench`: benchmark bodies and information measures

The code behind thread T12 of the roadmap. Each benchmark is *ready to go*: it has a model,
a simulator with a declared noise, a connection to the shared measures, a test, and one
baseline number (`python -m mibook.bench.baselines`). Studies are run when a question needs
them.

## Layers

1. **Bodies** (`bodies/`). Every benchmark registers a factory:
   `bench.make(name, **params)`, and `bench.available()` lists them. Linear bodies are
   `LinearBody` (mechanics H, D, K) or `StateSpaceBody` (mechanics plus first-order states
   such as aerodynamic lags). Nonlinear and contact bodies are `MechSim`, or a subclass that
   overrides `step` for impacts or time-dependent gaits. Every body names its environment
   channels (inputs such as forces and gusts), its parts (what a part of the body reads),
   and its parameters (properties such as stiffness or a contact point).
2. **Engines.**
   - `linear`: exact for linear bodies with Gaussian noise. Window measures use a Kalman
     whitening filter, so the cost grows linearly with the window; they reproduce the
     book's stacked computation to 1e-13. Spectral rates use frequency-domain formulas.
   - `sampling`: batched simulation (BAOAB, exact thermal velocity steps) for nonlinear
     and contact bodies, checked against the linear engine on linear bodies.
3. **Measures.**
   - Linear: `profile_bits`, `white_input_bits` (also the directed information and the
     accumulated transfer entropy, since nothing feeds back), `input_rate`, `fisher_rate`
     (thermal only, or through an input), `memory_capacity`, `mc_w`, `empowerment`, and
     `simulate_linear`.
   - Sampling: `lmmse_bits` (a lower bound for Gaussian inputs), `path_fisher` (Girsanov
     score, whole state observed), `feature_fisher` (a lower bound from summary features;
     works for any body and any parameter).

## Benchmarks

| Tier | Name | What it is |
|:--|:--|:--|
| linear | `rigid`, `sea` | the book's legs |
| linear | `network` | random mass-spring-damper network (morphological and reservoir computing) |
| linear | `whisker` | cantilever in a compliant follicle, touched along its length |
| nonlinear | `soft_arm` | ladder truss with stiffening springs (physical reservoir) |
| nonlinear | `bistable_chain` | double wells in series, driven at one end (hysterons) |
| contact | `hopper` | vertical spring-loaded hopper on a ground of unknown stiffness |
| contact | `gripper` | compliant pads squeezing an object of unknown stiffness |
| contact | `walker` | rimless wheel on a slope |
| contact | `tentacle` | soft arm curled by a tendon around an object |
| contact | `many_legged` | hyper-redundant robot whose stance legs find rough ground at random |
| fluid | `foil` | typical section, unsteady thin-airfoil aerodynamics, gusts and vortex streets |
| fluid | `bird` | body and wings joined by a compliant shoulder, in turbulence |
| collective | `swarm` | tethered agents coupled to neighbours, sensing a common force |
| linear | `pendulum` | inverted pendulum on a spring: stable or unstable by one parameter |

## Unstable bodies

An unstable body has no stationary law. Window measures take a declared `prior`
(`linear.prior_rest`); `linear.steady_white_rate` gives rates from the Riccati equation;
`linear.mirror_stabilize` closes the loop with the least-energy stabilizing feedback
(unstable eigenvalues reflected) and reports `control_power` and `min_data_rate`;
`sampling.mirror_feedback` does the same for a nonlinear body at an equilibrium. See the
research note of 2026-11-26.

## Video demos

`python -m mibook.bench.animate OUTDIR [name ...]` writes a 5 s MP4 per benchmark from a
real simulation, with display magnification and slow motion printed on the frame.

## Conventions and caveats

- The default noise is the body's own thermal noise through its dampers, at a declared
  `sigma_T2`. For bodies in a flow, turbulence and vortices are environment inputs, and the
  resolution comes from structural damping only (an open modeling choice, see T12).
- Contact damping is deterministic; only D carries thermal noise.
- In `MechSim`, D is constant (fixed at rest for the soft arm and the tentacle).
- Speed: about 2 microseconds per body-step for the 40-coordinate soft arm with numpy.
  Compiling the force kernels is the next step if a study needs more.
