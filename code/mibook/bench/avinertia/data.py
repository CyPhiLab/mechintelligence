# SPDX-License-Identifier: GPL-3.0-or-later
#
# Translated from AvInertia's analysis script AnalysisFunctions/run_script.R
# (https://github.com/charvey23/AvInertia, GPL-3); see model.py for the licence note.
"""Specimen data of Harvey et al. (2022) in the form AvInertia takes.

The dataset is figshare item 14935659 (collection doi:10.6084/m9.figshare.c.5503989):
two spreadsheets of measurements (bird_measurements_readytorun.xlsx,
feathertrackingsheet.xlsx), an ID file, and one file of wing shapes per specimen
(<date>_<species>_<BirdID>_transformed.csv, landmarks in metres with the origin at the
humeral head). `load_specimen` repeats run_script.R: unit conversion, renaming, the move
of the landmarks to the VRP, the filter on physically possible shapes, and the correction
of secondaries that would pass into the body.

A small subset ships with the code (`DATA`, see its README). Pass `root` to read the full
dataset instead; spreadsheets are read with a minimal reader, so no extra packages are
needed.
"""
from __future__ import annotations

import csv
import glob
import os
import re
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .model import POINTS, Specimen

DATA = Path(__file__).resolve().parent.parent / "data" / "avinertia"

SKIP = {("tyt_alb", "18_1"), ("fal_per", "18_1"), ("ard_her", "20_317")}
"""Specimens run_script.R removes for missing body data."""


# ---------------------------------------------------------------------------------------
# Reading tables
# ---------------------------------------------------------------------------------------

_NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
       "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships"}


def _col_index(ref: str) -> int:
    letters = re.match(r"[A-Z]+", ref).group(0)
    n = 0
    for ch in letters:
        n = 26 * n + ord(ch) - 64
    return n - 1


def read_xlsx(path, sheet: str) -> list[dict]:
    """Rows of one worksheet as dicts keyed by the header row. Numbers become floats,
    booleans bools, empty cells None."""
    with zipfile.ZipFile(path) as z:
        shared = []
        if "xl/sharedStrings.xml" in z.namelist():
            for si in ET.fromstring(z.read("xl/sharedStrings.xml")).findall("m:si", _NS):
                shared.append("".join(t.text or "" for t in si.iter(f"{{{_NS['m']}}}t")))
        wb = ET.fromstring(z.read("xl/workbook.xml"))
        rid = next(s.get(f"{{{_NS['r']}}}id") for s in wb.find("m:sheets", _NS)
                   if s.get("name") == sheet)
        rels = ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))
        target = next(r.get("Target") for r in rels if r.get("Id") == rid)
        target = target.lstrip("/")
        target = target if target.startswith("xl/") else "xl/" + target
        root = ET.fromstring(z.read(target))
    rows = []
    for row in root.find("m:sheetData", _NS).findall("m:row", _NS):
        cells = {}
        for c in row.findall("m:c", _NS):
            t, v = c.get("t"), c.find("m:v", _NS)
            if t == "inlineStr":
                val = "".join(x.text or "" for x in c.iter(f"{{{_NS['m']}}}t"))
            elif v is None:
                val = None
            elif t == "s":
                val = shared[int(v.text)]
            elif t == "b":
                val = v.text == "1"
            elif t in ("str", "e"):
                val = v.text
            else:
                val = float(v.text)
            cells[_col_index(c.get("r"))] = val
        rows.append(cells)
    header = rows[0]
    names = [header.get(i) for i in range(max(header) + 1)]
    out = []
    for cells in rows[1:]:
        if not any(v not in (None, "") for v in cells.values()):
            continue
        out.append({n: cells.get(i) for i, n in enumerate(names) if n is not None})
    return out


_NUMBER = re.compile(r"[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?")


def _num(v):
    if v is None or v == "" or v == "NA":
        return np.nan
    if isinstance(v, bool):
        return v
    if isinstance(v, str) and not _NUMBER.fullmatch(v.strip()):
        return v                                     # e.g. BirdID "20_0341", not a number
    return float(v)


def read_csv(path) -> list[dict]:
    with open(path, newline="") as f:
        return [{k: _num(v) for k, v in r.items()} for r in csv.DictReader(f)]


def write_csv(path, rows: list[dict], columns=None):
    columns = columns or list(rows[0])
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if (isinstance(v, float) and np.isnan(v)) else v)
                        for k, v in r.items() if k in columns})


@dataclass
class Tables:
    body: list[dict]          # sheet 'Major body parts'
    wing: list[dict]          # sheet 'Within wings'
    feathers: list[dict]      # sheet 'FeatherData'
    barbs: list[dict]         # sheet 'BarbInfo'
    ids: list[dict]           # IDfile.csv
    wing_dir: Path


def load_tables(root=None) -> Tables:
    """The measurement tables, from the full dataset folder `root` (spreadsheets) or from
    the shipped subset (CSV copies of the same sheets)."""
    if root is None:
        d = DATA
        return Tables(read_csv(d / "body.csv"), read_csv(d / "wing.csv"),
                      read_csv(d / "feathers.csv"), read_csv(d / "barbs.csv"),
                      read_csv(d / "ids.csv"), d / "wings")
    r = Path(root)
    meas, feat = r / "bird_measurements_readytorun.xlsx", r / "feathertrackingsheet.xlsx"
    clean = lambda rows: [{k: _num(v) if not isinstance(v, str) else v for k, v in x.items()}
                          for x in rows]
    return Tables(clean(read_xlsx(meas, "Major body parts")), clean(read_xlsx(meas, "Within wings")),
                  clean(read_xlsx(feat, "FeatherData")), clean(read_xlsx(feat, "BarbInfo")),
                  read_csv(r / "IDfile.csv"), r)


# ---------------------------------------------------------------------------------------
# run_script.R
# ---------------------------------------------------------------------------------------

def split_id(bird_id: str) -> tuple[str, str]:
    """Species code and BirdID from a spreadsheet bird_id, as in run_script.R."""
    p = bird_id.split("_")
    if p[0] == "COLLI":
        return "col_liv", f"{p[1]}_{p[2]}"
    return f"{p[0].lower()}_{p[1]}", f"{p[2]}_{p[3]}"


_RENAME = {
    "humerus_muscles_mass_g": "brachial_muscle_mass", "raduln_muscles_mass_g": "antebrachial_muscle_mass",
    "cmc_muscles_mass_g": "manus_muscle_mass", "whole_body_mass_g": "total_bird_mass",
    "tert_feathers_mass_g": "tertiary_mass", "skin_coverts_mass_g": "all_skin_coverts_mass",
    "head_length_cm": "head_length", "head_mass_g": "head_mass", "head_height_max_cm": "head_height",
    "neck_mass_g": "neck_mass", "neck_width_cm": "neck_width", "neck_length_cm": "neck_length",
    "torsotail_length_cm": "torsotail_length", "torsotail_mass_g": "torsotail_mass",
    "tail_length_cm": "tail_length", "tail_width_cm": "tail_width", "right_leg_mass_g": "right_leg_mass",
    "body_width_max_cm": "body_width_max", "body_height_max_cm": "body_height_max",
    "width_at_leg_insert_cm": "body_width_at_leg_insert", "x_loc_of_body_max_cm": "x_loc_of_body_max",
    "x_loc_leg_insertion_cm": "x_loc_leg_insertion", "x_loc_TorsotailCoG_cm": "x_loc_TorsotailCoG",
    "z_loc_TorsotailCoG_cm": "z_loc_TorsotailCoG", "x_loc_of_humeral_insert_cm": "x_loc_humeral_insert",
    "y_loc_of_humeral_insert_cm": "y_loc_humeral_insert", "z_loc_of_humeral_insert_cm": "z_loc_humeral_insert",
}
# run_script.R renames "left_leg_mass_" (no match) and never renames tail_mass_g; R's
# partial matching of `$` then reads left_leg_mass_g and tail_mass_g. The aliases below do
# the same explicitly.
_ALIAS = {"left_leg_mass": "left_leg_mass_g", "tail_mass": "tail_mass_g"}


def _si(name: str, v):
    if not isinstance(v, float) or np.isnan(v):
        return v
    if "mass_g" in name:
        return round(v / 1000, 7)
    if "_cm" in name:
        return round(v / 100, 4)
    if "_mm" in name:
        return round(v / 1000, 4)
    return v


def bird_rows(t: Tables) -> dict[tuple[str, str], dict]:
    """The merged body, wing and barb measurements of every specimen, in SI units, keyed by
    (species, BirdID)."""
    wing = {}
    for r in t.wing:
        wing[split_id(r["bird_id"])] = r
    barbs = {r["species"]: r for r in t.barbs}
    out = {}
    for r in t.body:
        key = split_id(r["bird_id"])
        if key not in wing or key[0] not in barbs:
            continue
        merged = {**{k: v for k, v in r.items()}, **{k: v for k, v in wing[key].items()}}
        b = {}
        for k, v in merged.items():
            b[_RENAME.get(k, k)] = _si(k, v)
        for a, k in _ALIAS.items():
            b[a] = b[k]
        b["x_loc_TorsotailCoG"] = -b["x_loc_TorsotailCoG"]
        b["z_loc_humeral_insert"] -= b["z_dist_to_veh_ref_point_cm"]
        b["z_loc_TorsotailCoG"] -= b["z_dist_to_veh_ref_point_cm"]
        b["wing_mass"] = b["right_wing_mass_g"] if b["right_or_left"] == "Right" else b["left_wing_mass_g"]
        b["barb_distance"] = barbs[key[0]]["barb_distance_um"] * 1e-6
        b["barb_radius"] = barbs[key[0]]["barb_radius_um"] * 1e-6
        ext = b["extend_neck"]
        b["extend_neck"] = ext if isinstance(ext, bool) else str(ext).upper() == "TRUE"
        b["species"], b["BirdID"] = key
        out[key] = b
    return out


def species_feathers(t: Tables, species: str) -> tuple[list[dict], str]:
    """Mean feather geometry of a species from the feather tracking sheet, with the angle
    conventions of run_script.R. Returns the feathers (lengths in m) and the BirdID of the
    specimen they were measured on."""
    rows = [r for r in t.feathers if r["species"] == species]
    groups: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        r = dict(r)
        if r["Component"] == "calamus length" and r["Angle"] < 0:
            r["Angle"] = 180 + r["Angle"]
        if r["Component"] == "vane length" and r["Angle"] > 0:
            r["Angle"] = r["Angle"] - 180
        groups.setdefault((r["Feather"], r["Component"]), []).append(r)
    mean = lambda f, c, q: (float(np.mean([g[q] for g in groups[(f, c)]]))
                            if (f, c) in groups else np.nan)
    names = sorted({f for f, _ in groups})
    have_area = {f for f, c in groups if c in ("distal vane", "proximal vane")}
    have_len = {f for f, c in groups if c in ("calamus length", "vane length", "rachis width")}
    left = rows[0]["right_or_left"] == "Left"
    feathers = []
    for f in names:
        if f not in have_area or f not in have_len:
            continue
        lv, lc = mean(f, "vane length", "Length"), mean(f, "calamus length", "Length")
        va = 180 - (mean(f, "calamus length", "Angle") - mean(f, "vane length", "Angle"))
        feathers.append(dict(feather=f, l_vane=0.01 * lv, l_cal=0.01 * lc,
                             w_cal=0.01 * mean(f, "rachis width", "Length"),
                             w_vd=0.01 * mean(f, "distal vane", "Area") / lv,
                             w_vp=0.01 * mean(f, "proximal vane", "Area") / lv,
                             vane_angle=-va if left else va))
    bid = rows[0]["bird_id"]
    return feathers, bid if isinstance(bid, str) else str(bid)


def load_specimen(species: str, bird_id: str, tables: Tables | None = None,
                  birds: dict | None = None) -> Specimen:
    """AvInertia's inputs for one specimen (run_script.R, per-specimen block)."""
    t = tables or load_tables()
    birds = birds or bird_rows(t)
    b = birds[(species, bird_id)]
    feathers, ref_id = species_feathers(t, species)
    ref = birds.get((species, ref_id))
    out = []
    for f in feathers:
        f = dict(f)
        key = f["feather"].lower() + "_mass_g"
        f["m_f"] = b[key]
        if ref_id != bird_id:
            s = (b[key] / ref[key]) ** (1 / 3)
            for q in ("l_vane", "l_cal", "w_cal", "w_vp", "w_vd"):
                f[q] = s * f[q]
        out.append(f)
    out.append(dict(feather="alula", m_f=b["alular_mass_g"]))
    bones = {}
    for name, col in (("Humerus", "humerus"), ("Ulna", "ulna"), ("Radius", "radius"),
                      ("Carpometacarpus", "cmc")):
        r_out = 0.5 * b[f"{col}_diameter_mm"]
        bones[name] = (b[f"{col}_mass_g"], b[f"{col}_length_mm"], r_out, 0.78 * r_out)
    bones["Ulnare"] = (b["ulnare_mass_g"],)
    bones["Radiale"] = (b["radiale_mass_g"],)
    return Specimen(species, bird_id, b, bones, out)


def specimens(tables: Tables | None = None) -> list[tuple[str, str]]:
    """The specimens the paper analysed: those in the ID file with body data and
    use_body == 'Y', sorted by species."""
    t = tables or load_tables()
    birds = bird_rows(t)
    keys = sorted({(r["species"], str(r["BirdID"])) for r in t.ids} - SKIP)
    return [k for k in keys if k in birds and birds[k]["use_body"] == "Y"]


@dataclass
class WingShapes:
    """Wing configurations of one specimen after run_script.R's processing. `pts` maps
    'pt1'..'pt12' to arrays (N, 3) in the VRP frame; `clean` is (N, 10, 3) in the order of
    model.POINTS."""
    test_id: np.ndarray
    frame_id: np.ndarray
    elbow: np.ndarray
    manus: np.ndarray
    pts: dict
    S_proj: np.ndarray
    S: np.ndarray

    @property
    def clean(self) -> np.ndarray:
        names = [n.lower() for n in POINTS]
        return np.stack([self.pts[n] for n in names], axis=-2)

    def __len__(self):
        return len(self.elbow)


def wing_file(wing_dir, species: str, bird_id: str) -> Path:
    hits = sorted(glob.glob(os.path.join(str(wing_dir), f"*{species}_{bird_id}*")))
    if len(hits) != 1:
        raise FileNotFoundError(f"expected one wing file for {species} {bird_id}, found {hits}")
    return Path(hits[0])


def load_wings(sp: Specimen, tables: Tables | None = None, path=None) -> WingShapes:
    """The specimen's wing shapes, moved to the VRP and cleaned as in run_script.R."""
    t = tables or load_tables()
    path = path or wing_file(t.wing_dir, sp.species, sp.bird_id)
    rows = read_csv(path)
    b = sp.bird
    get = lambda k: np.array([r[k] for r in rows], float)
    pts = {}
    for i in (1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12):
        pts[f"pt{i}"] = np.stack([get(f"pt{i}_X") - b["x_loc_humeral_insert"],
                                  get(f"pt{i}_Y") + b["y_loc_humeral_insert"],
                                  get(f"pt{i}_Z") + b["z_loc_humeral_insert"]], axis=-1)
    y = {k: v[:, 1] for k, v in pts.items()}
    keep = ((y["pt2"] > 0) & (y["pt3"] > y["pt12"]) & (y["pt4"] > y["pt12"])
            & (y["pt6"] > y["pt12"]) & (y["pt7"] > y["pt12"]))
    pts = {k: v[keep].copy() for k, v in pts.items()}
    p1, p2 = pts["pt1"], pts["pt2"]
    # S1 tip inside the body: rotate it out to 5 mm beyond the shoulder (same distance to
    # the elbow in the x-y plane)
    p10 = pts["pt10"]
    bad = p10[:, 1] < p1[:, 1]
    yt = p1[:, 1] + 0.005
    with np.errstate(invalid="ignore"):              # R gives NaN here too
        xn = p2[:, 0] - np.sqrt((p10[:, 1] - p2[:, 1]) ** 2 + (p10[:, 0] - p2[:, 0]) ** 2
                                - (yt - p2[:, 1]) ** 2)
    p10[bad, 0], p10[bad, 1] = xn[bad], yt[bad]
    p11 = pts["pt11"]
    bad = p11[:, 1] < p1[:, 1]
    with np.errstate(invalid="ignore"):
        xn = p2[:, 0] - np.sqrt((p11[:, 1] - p2[:, 1]) ** 2 + (p11[:, 0] - p2[:, 0]) ** 2
                                - (p1[:, 1] - p2[:, 1]) ** 2)
    p11[bad, 0], p11[bad, 1] = xn[bad], p1[bad, 1]
    sel = lambda k: get(k)[keep]
    tid = np.array([r["TestID"] for r in rows])[keep]
    return WingShapes(tid, sel("FrameID"), sel("elbow"), sel("manus"), pts, sel("S_proj"), sel("S"))
