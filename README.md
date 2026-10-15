# Polynesia–South America Voyage Simulation

Simulated voyages to test which travel scenario best explains the
Polynesian–Native American contact around 1200 CE: Polynesians sailing to South
America and back, or South Americans reaching eastern Polynesia.

- **[SPEC.md](SPEC.md)**: the scientific design: questions, data, model, grid, validation, open questions.
- **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)**: module contract and shared conventions.

## Status

Milestone 1 (core) is built and tested on simple made-up winds and currents.
Real wind and current data come next (milestone 2 starts with downloading them
and confirming the validation voyages' figures).

| Piece | State |
| --- | --- |
| Wind and current interpolation; pre-1993 current climatology; last-millennium anomalies as added fields | done, tested |
| Vessels: speed tables, closest angle to the wind, calms and gales, SAROPS leeway; every setting sourced in `configs/vessels.yaml` | done, tested; canoe and raft speeds provisional |
| Strategies: drift, downwind, hold course, wait for wind, latitude sailing, westerlies route | done, tested |
| Land: GSHHG shorelines grouped into island groups and coast segments, heights from literature and the Copernicus DEM | done, tested on the real shoreline data |
| Landfall: sighting range by height plus seabird range, whole-path test, only destinations end a voyage | done, tested |
| Integrator: vectorized steps, per-voyage random streams, one run for every supply limit and sighting rule | done, tested |
| El Niño labels (RONI, ONI, HadISST), success probabilities, July–June year bootstrap | done, tested |
| Data download (ERA5, GLORYS, GSHHG, El Niño indices) | written and tested offline; needs free accounts to run |
| One command per scenario grid, byte-identical reruns | done, tested |
| Validation voyages, full grid, round trips, maps, report | milestones 2–5 |

## Setup

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"          # core + tests
pip install -e ".[fetch]"        # data download clients (optional)
python -m pytest
```

## Running

```bash
voyage-sim run configs/synthetic_demo.yaml    # made-up winds, no downloads; about 10 s
```

Each run writes `outputs/<name>/`: one `.npz` of voyage results per grid cell, a
`summary.csv` of success probabilities with intervals and durations, and the
resolved config. Reruns give byte-identical files. The demo's numbers exercise
the pipeline; they are not results.

## Data

```bash
voyage-sim fetch gshhg --out data/gshhg            # shorelines (public, checksummed)
voyage-sim fetch enso --out data/enso              # RONI, ONI, HadISST Niño-3.4
voyage-sim fetch era5 --years 2010-2019 --out data/era5 --dry-run
voyage-sim fetch glorys --start 2010-01-01 --end 2019-12-31 --out data/glorys --dry-run
```

ERA5 needs a free [Copernicus Climate Data Store](https://cds.climate.copernicus.eu)
account (personal access token in `~/.cdsapirc`). GLORYS needs a free
[Copernicus Marine](https://data.marine.copernicus.eu) account
(`copernicusmarine login`). `configs/coarse_decade.yaml` runs the spec's coarse
grid once 2010–2019 is downloaded.
