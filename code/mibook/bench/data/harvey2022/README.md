# Gull glider data for `mibook.bench.bodies.glider`

From Harvey and Inman (2022), "Gull dynamic pitch stability is controlled by wing morphing",
PNAS 119: e2204847119, data collection
[doi:10.6084/m9.figshare.c.5895032](https://doi.org/10.6084/m9.figshare.c.5895032), and
code github.com/charvey23/AvianDynStab (commit 5f64416).

| File | Source | Content |
|---|---|---|
| `2022_02_07_coefficients_all.csv` | the figshare collection, unchanged | fitted models for Iyy, CL, CD, Cm, dCL/dq and dCm/dq as functions of elbow, wrist (manus), shoulder sweep and dihedral angles, angle of attack (deg) and lift coefficient; written by AvianDynStab's `determine_functions.R` |
| `gull_joint_hull.csv` | computed here | the 47 vertices, ordered by angle about their mean, of the alpha shape (alpha = 30 deg) of the gull's measured elbow and wrist angles; see below |

`gull_joint_hull.csv` reproduces AvianDynStab's `cut_trueshape` (`support_functions.R`),
which keeps only elbow and wrist angles inside the alpha hull of the wing shapes of the
glaucous-winged gull (`lar_gla`, shoulder sweep and dihedral 0) in
`2020_05_25_OrientedWings.csv` from the data of Harvey et al. (2021), J. R. Soc. Interface
18: 20210132 (figshare item 13615913). The 1,499 distinct (elbow, wrist) pairs were taken
from that file; a Delaunay edge is on the alpha shape when one of the two discs of radius 30
through its ends holds no other point (the definition used by R's `alphahull`). With this
polygon, the configurations in the paper's own results file
(`2022_02_07_LongDynStability_Rigid.csv`) give exactly the counts and ranges printed in the
paper (1,457 trimmed, 1,331 stable, and the ranges tested in `tests/test_bench_glider.py`).

**Licence.** The repositories carry no licence; the book's author has permitted porting
code and data from them. The port in `glider.py` is distributed under GPL-3.0-or-later like
the AvInertia port. Cite the paper and the figshare collection wherever these data are used.
