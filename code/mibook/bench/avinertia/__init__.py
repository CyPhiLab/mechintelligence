# SPDX-License-Identifier: GPL-3.0-or-later
"""A Python translation of AvInertia (Harvey et al. 2022, Nature 603, 648-653): the centre
of gravity and moment of inertia tensor of a bird for any elbow and wrist configuration, and
the paper's estimates of static margin and pitch agility. Inputs for the bird glider
benchmark.

This subpackage is a derivative of AvInertia (GPL-3, https://github.com/charvey23/AvInertia)
and is distributed under GPL-3.0-or-later; see model.py.

    from mibook.bench import avinertia as av
    sp = av.load_specimen("lar_gla", "20_0341")      # glaucous-winged gull
    w = av.load_wings(sp)                             # its wing shapes, VRP frame
    bird = av.bird_inertia(sp, w.clean)               # I, CG for every shape
    st = av.stability(bird, w)                        # static margin, agility
"""
from .data import (DATA, load_specimen, load_tables, load_wings, specimens, read_xlsx,
                   bird_rows, WingShapes)
from .model import (Bird, Mass, Specimen, bird_inertia, combine_inertialprop,
                    compute_feat_inertia, massprop_birdwing, massprop_restbody, shift_origin,
                    POINTS, DENSITY)
from .stability import Stability, stability, standard_mean_c4, root_chord

__all__ = ["DATA", "load_specimen", "load_tables", "load_wings", "specimens", "read_xlsx",
           "bird_rows", "WingShapes", "Bird", "Mass", "Specimen", "bird_inertia",
           "combine_inertialprop", "compute_feat_inertia", "massprop_birdwing",
           "massprop_restbody", "shift_origin", "POINTS", "DENSITY", "Stability",
           "stability", "standard_mean_c4", "root_chord"]
