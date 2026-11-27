# Bird specimen data for `mibook.bench.avinertia`

A small subset of the specimen data of Harvey, Baliga, Wong, Altshuler and Inman (2022),
"Birds can transition between stable and unstable states via wing morphing", Nature 603,
648–653, enough for the fast tests and for the gull glider benchmark. The full dataset is
figshare item 14935659, part of collection
[doi:10.6084/m9.figshare.c.5503989](https://doi.org/10.6084/m9.figshare.c.5503989).
To use all of it, unpack it and pass its folder as `root` to `load_tables`, or set
`AVINERTIA_DATA` for the tests.

**Licence: not confirmed.** Figshare's default for such items is CC BY 4.0, but the item
page could not be reached from the environment where this subset was made, so the licence
has not been checked. Until it is, keep this subset minimal and cite the paper and the
figshare collection wherever the data are used.

| File | Source | Content |
|---|---|---|
| `body.csv` | `bird_measurements_readytorun.xlsx`, sheet "Major body parts" | all rows and columns, as in the spreadsheet |
| `wing.csv` | same file, sheet "Within wings" | all rows and columns |
| `feathers.csv` | `feathertrackingsheet.xlsx`, sheet "FeatherData" | rows for the pigeon (`col_liv`) and the glaucous-winged gull (`lar_gla`) only |
| `barbs.csv` | same file, sheet "BarbInfo" | all rows |
| `ids.csv` | `IDfile.csv` | rows for pigeon 20_0300 and gull 20_0341 |
| `wings/*_subset.csv` | `2021_05_13_<species>_<BirdID>_transformed.csv` | every 8th wing shape (plus pigeon test 4, frame 317), columns `time_sec`, `wing`, `elbow_round`, `manus_round` dropped; values copied unchanged |
| `package_example/` | AvInertia 0.0.2, `data/*.rda` (GPL-3) | the package's example inputs (pigeon 20_0300, test 4, frame 317), exported to CSV |
| `plot_masses.csv` | supplied by the author (`bird_masses_2022.csv`) | whole-bird masses read from the paper's plots |

The spreadsheets were converted to CSV with `mibook.bench.avinertia.read_xlsx` and
`data.write_csv`; numbers are written at full precision.
