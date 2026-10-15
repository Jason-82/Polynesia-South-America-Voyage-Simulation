# Polynesia–South America Voyage Simulation: Project Spec

Oct 15, 2026 · @Jason Abrams

## Goal and success criteria

Use simulated voyages to test which travel scenario best explains the Polynesian–Native American contact around 1200 CE: Polynesians sailing to South America and back, or South Americans reaching eastern Polynesia.

Genetics can date and place the contact but can't say who traveled. A simulation can't prove that either. What it can show is which directions, routes, seasons and climate states make each scenario feasible, how long each takes, and where landfalls concentrate.

The project succeeds when all four hold:

- **Calibrated:** the model reproduces documented modern voyages and the main results of earlier simulations before any new scenario is trusted.
- **Comparative:** it reports success odds and durations for both directions, under drifting and under deliberate sailing, on the same footing.
- **Specific:** it names the months and climate states (El Niño, La Niña, neutral) that raise or lower the odds.
- **Useful to fieldwork:** it maps where successful landfalls concentrate on each coast, so archaeologists know where to look.

## What's known

The contact itself is settled; who traveled, how many times, and from which coast are not, and those are exactly the questions a simulation can narrow.

| Evidence | Finding | What it means for the simulation |
| --- | --- | --- |
| [Ioannidis et al. 2020](https://www.nature.com/articles/s41586-020-2487-2), modern genomes | 807 people from 17 islands and 15 Native American groups; one contact event in eastern Polynesia around 1200 CE, with a group closest to the Indigenous people of present-day Colombia | Test the Marquesas and the Colombian–Ecuadorian coast first |
| Same study, per-island dates as summarized in [Moreno-Mayar et al. 2024](https://www.nature.com/articles/s41586-024-07881-4) | Admixture dated from about 1150 CE (South Marquesas) to 1380 CE (Rapa Nui) | South Marquesas is the earliest landing to explain |
| [Moreno-Mayar et al. 2024](https://www.nature.com/articles/s41586-024-07881-4), 15 ancient Rapanui genomes | About 10% Native American ancestry, mixed in about 1250–1430 CE; the source looks closest to Central Andean populations; contact may have happened more than once, and its direction is still unresolved | Add Rapa Nui and the Peruvian coast as a second scenario |
| [Ioannidis et al. 2021](https://www.immunology.ox.ac.uk/publications/1195670), settlement sequence | Society Islands settled in the 11th century, Tuamotus in the 12th, then the Marquesas, Raivavae and Rapa Nui around 1200 CE via Mangareva | Limits which islands could launch voyages, and when |
| Sweet potato and bottle gourd, noted in [Ioannidis et al. 2020](https://www.nature.com/articles/s41586-020-2487-2) | American crops in the Polynesian archaeological record before Europeans arrived | A plant transfer that drift alone might explain |

Earlier simulations to build on:

- **1973 drift study.** Levison, Ward and Webb ran 101,016 simulated drift voyages; with few exceptions, drifting could not explain the settlement of Polynesia from either Melanesia or South America ([summary](https://tnm.journals.yorku.ca/index.php/default/article/download/628/602)).
- **Montenegro et al. 2008.** Drift from 160 launch sites on the American coast; sweet potato seed pods could reach seven island groups, with the best odds in the Marquesas. The lead author still judged Polynesians sailing to South America the likeliest route ([Nature news](https://www.nature.com/news/2007/070514/full/news070514-20.html)).
- **Montenegro, Callaghan and Fitzpatrick 2016.** High-resolution wind and current data plus El Niño effects; success varied strongly by season and climate state, and Rapa Nui was reachable from Mangareva or the Marquesas ([abstract](https://repositorio.unesp.br/handle/11449/162171)).
- **The windward debate.** Anderson et al. argue early canoes could not sail upwind and that settlement pulses match periods of El Niño westerlies ([abstract](https://openresearch-repository.anu.edu.au/items/150a3fde-71f1-452a-835d-12b93b8f840b)); Finney's reconstructed canoes sail about 75° off the wind ([Finney](https://lir.byuh.edu/index.php/pacific/article/download/190/181/341)). The simulation should run both assumptions.

The gap: the research for this spec found no simulation that tests both directions against the 2020 and 2024 genetic results, including the return trip a Polynesian round voyage requires.

## Data sources

All inputs are free with a registered account: modern winds and currents for the main runs, last-millennium climate runs to check that 1200 CE conditions don't change the answer, and documented voyages for calibration.

| Source | What it provides | Coverage | Access |
| --- | --- | --- | --- |
| [ERA5 hourly, single levels](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels) | 10 m winds and wave fields | 1940 to present; hourly; 0.25° grid (waves 0.5°) | Copernicus Climate Data Store API, free account, CC-BY |
| [GLORYS12V1 ocean reanalysis](https://data.marine.copernicus.eu/product/GLOBAL_MULTIYEAR_PHY_001_030/description) | Daily surface currents | 1993 onward; 1/12° (about 8 km) | Copernicus Marine Toolbox, free account |
| [PMIP4 past1000 runs](https://gmd.copernicus.org/articles/10/4005/2017/gmd-10-4005-2017.html) | Climate-model simulations of 850–1849 CE | Varies by model; daily wind output must be checked per model | ESGF data portals |
| [Hōkūleʻa's 1976 voyage](https://mauinow.com/2026/04/24/50th-anniversary-of-the-historic-launch-of-hokule%CA%BBas-1976-maiden-voyage/) | Calibration: Hawaiʻi to landfall at Mataiva in 31 days; Tahiti to Oʻahu in 22 days | One documented round trip | Published timeline |
| [Hōkūleʻa specifications](https://en.wikipedia.org/wiki/H%C5%8Dk%C5%ABle%CA%BBa) | Canoe speed of 4–6 knots fully laden | — | Public |
| Kon-Tiki, 1947 | Calibration for a sailing raft: Peru to the Tuamotus in 101 days | One voyage | Public record |
| Balsa-raft engineering (Dewan and Hosler 2008, cited in [Ioannidis et al. 2020](https://www.nature.com/articles/s41586-020-2487-2)) | Performance limits for South American sailing rafts | — | Journal article |

Notes for the builder:

- **Currents before 1993.** GLORYS starts in 1993, so wind years from 1940 to 1992 pair with a monthly current climatology built from GLORYS. Flag those runs in the output.
- **El Niño labels.** Classify each modern year as El Niño, La Niña or neutral with NOAA's Oceanic Niño Index, so results can be split by climate state.
- **Coastlines.** Use a high-resolution shoreline (GSHHG) for islands and the American coast; small islands matter.

## Model design

&#91;embedded content: voyage simulator · one voyage, two decisions per step\]

Each voyage advances in 6-hour steps; thousands run in parallel, and only the statistics of their outcomes matter.

Vessel settings (starting values, adjusted during validation):

| Vessel | Speed | Closest angle to the wind | Basis |
| --- | --- | --- | --- |
| Double canoe, upwind-capable | 4–6 knots typical | About 75° | Hōkūleʻa; Finney |
| Double canoe, downwind only | 4–6 knots typical | 90°, no upwind sailing | Tests Anderson et al.'s hypothesis |
| Sailing balsa raft | Set from Dewan and Hosler 2008 | Set from the same source | Open question |
| Drifting craft | Current plus a fraction of wind speed (leeway) | No steering | Standard drift modeling |

Strategy rules:

- **Drift:** no steering; the craft moves with the current and the wind's push.
- **Downwind:** sail only within the vessel's downwind range.
- **Hold course:** steer for a target; when it lies closer to the wind than the vessel can sail, tack.
- **Wait for wind:** stay in port until winds favor the target, then leave.
- **Latitude sailing:** reach the target's latitude, then run east or west along it.

A voyage succeeds when it comes within sighting range of a shoreline (start at 40 km, test 20–60 km). It fails when days at sea exceed the supply limit.

## Experiments

Run a grid of scenarios, launch many voyages per cell with randomized start days, and compare success odds and durations across the grid.

| Dimension | Values |
| --- | --- |
| Direction | Polynesia to South America and back; South America to Polynesia |
| Polynesian departures | South Marquesas (Fatu Hiva), North Marquesas, Mangareva, Tuamotus; Rapa Nui only for dates after its settlement |
| American coast segments | Colombia, Ecuador, northern Peru, southern Peru, northern Chile |
| Season | Every month |
| Climate state | Modern years labeled El Niño, La Niña or neutral; last-millennium model years 1100–1300 CE |
| Vessel | Double canoe sailing 75° off the wind; double canoe sailing downwind only; sailing balsa raft; unpowered drift |
| Strategy | Drift; run downwind; hold a course; wait in port for favorable winds; latitude sailing |
| Supplies | 30, 60 or 90 days at sea |

For every cell, report:

- Success odds with 95% bootstrap intervals.
- Median and 90th-percentile days at sea.
- A landfall density map for the receiving coast.
- For round trips, the joint odds of going and returning, allowing a stay of one to six months and the best return month.

The four questions the grid has to answer:

1. Can a Marquesas canoe reach the Colombian–Ecuadorian coast and get back, and in which months and climate states?
2. Can a raft from that coast, drifting or sailing, reach the South Marquesas, where the earliest admixture appears?
3. Does a Rapa Nui or Mangareva route to Peru work better for the later, Andean-linked contact?
4. How much do the answers change if canoes could not sail upwind?

Start with a coarse grid (one modern decade, two vessels, two strategies) to debug before scaling up.

## Validation

Trust the model on new scenarios only after it reproduces documented voyages and earlier published results.

| Check | Target | Tolerance |
| --- | --- | --- |
| Hōkūleʻa 1976, southbound | Maui (left May 1) to first landfall at Mataiva: 31 days | Duration within ±20%; landfall in the Tuamotus |
| Hōkūleʻa 1976, northbound | Tahiti (left July 4) to Oʻahu: 22 days | Duration within ±20% |
| Kon-Tiki 1947 | Peru to the Tuamotus: 101 days, with raft settings | Duration within ±25%; landfall in the Tuamotus |
| Montenegro et al. 2008 | Drift from the American coast lands most often in the Marquesas | Same top-ranked island group |
| Montenegro et al. 2016 | Rapa Nui reachable from Mangareva or the Marquesas, with strong seasonal and El Niño effects | Same qualitative pattern |

Then stress-test the results:

- **Time step:** 1 hour versus 6 hours.
- **Sighting radius:** 20, 40 and 60 km.
- **Leeway:** ±50%.
- **Canoe speed:** ±20%.

Any headline result that flips under these changes is reported as fragile.

## Deliverables, repo layout, and acceptance criteria

The specialist hands back a tested simulator, the scenario results, and a short report with maps; one command per scenario grid reruns everything.

```
voyage-sim/
  configs/          # one YAML per scenario grid: departures, months, vessels, strategies
  src/
    fetch.py        # ERA5 via the CDS API; GLORYS via the Copernicus Marine Toolbox; GSHHG shorelines
    fields.py       # interpolate wind and current in space and time; pre-1993 current climatology
    vessels.py      # speed tables, leeway, upwind limit per vessel type
    strategies.py   # drift, downwind, hold course, wait for wind, latitude sailing
    simulate.py     # vectorized: many voyages per step, fixed random seeds
    landfall.py     # sighting-radius test against shorelines and islands
    analyze.py      # odds, durations, bootstrap intervals, round-trip pairing
    plots.py        # track bundles and landfall density maps
  tests/
  outputs/          # git-ignored
  REPORT.md
```

Acceptance criteria:

- [ ] All five validation checks pass, or the report explains each miss
- [ ] Every scenario grid reruns from its config with identical results
- [ ] The report gives success odds and durations for both directions, by vessel, strategy, month and climate state, with intervals
- [ ] Landfall density maps exist for both coasts
- [ ] A parameter table lists every vessel and strategy setting with its source
- [ ] Sensitivity results cover time step, sighting radius, leeway and speed

Check with the project owner before: buying cloud compute, contacting researchers, or publishing anything.

## Milestones, risks, and open questions

A validated simulator is about a week of part-time work; the full grid and report take two to three weeks more.

1. **Core (days 1–3):** data access, wind and current interpolation, one vessel type, the integrator.
2. **Validation (days 4–6):** the five checks and the stress tests.
3. **Grid (week 2):** the full scenario grid on modern climate.
4. **Past climate (week 3):** rerun the key cells on last-millennium model years.
5. **Report (weeks 3–4):** odds tables, maps, and a plain-language summary.

| Risk | Early signal | Fallback |
| --- | --- | --- |
| Modern winds differ from 1200 CE climate | Last-millennium runs disagree with modern results | Report both; treat agreement as robust and disagreement as uncertain |
| Vessel performance is uncertain | Validation durations miss their tolerance | Calibrate speed tables to Hōkūleʻa; report results as ranges |
| The sighting rule drives the answer | Odds swing widely across sighting radii | Report the spread rather than one number |
| Human strategy is hard to model | Results depend mostly on the strategy chosen | Bracket with simple strategies; don't claim a single best route |
| Last-millennium daily winds unavailable | Model archives hold only monthly winds | Add modern day-to-day variability to the monthly means |

Open questions for the owner:

- [ ] Which raft performance numbers to adopt from Dewan and Hosler 2008?
- [ ] Include a one-way scenario where South Americans reach Polynesia and stay?
- [ ] Ask a Pacific voyaging researcher to review the strategy rules before the full grid?
- [ ] Publish the results, or keep them as a personal study?

Sources: [Ioannidis et al. 2020](https://www.nature.com/articles/s41586-020-2487-2) · [Moreno-Mayar et al. 2024](https://www.nature.com/articles/s41586-024-07881-4) · [Ioannidis et al. 2021](https://www.immunology.ox.ac.uk/publications/1195670) · [Montenegro et al. 2016](https://repositorio.unesp.br/handle/11449/162171) · [Nature news on Montenegro et al. 2008](https://www.nature.com/news/2007/070514/full/news070514-20.html) · [Computer simulations of ancient voyaging](https://tnm.journals.yorku.ca/index.php/default/article/download/628/602) · [Anderson et al. 2006](https://openresearch-repository.anu.edu.au/items/150a3fde-71f1-452a-835d-12b93b8f840b) · [Finney on canoe performance](https://lir.byuh.edu/index.php/pacific/article/download/190/181/341) · [ERA5](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels) · [GLORYS12V1](https://data.marine.copernicus.eu/product/GLOBAL_MULTIYEAR_PHY_001_030/description) · [PMIP4 past1000](https://gmd.copernicus.org/articles/10/4005/2017/gmd-10-4005-2017.html) · [Hōkūleʻa 1976 timeline](https://mauinow.com/2026/04/24/50th-anniversary-of-the-historic-launch-of-hokule%CA%BBas-1976-maiden-voyage/) · [Hōkūleʻa specifications](https://en.wikipedia.org/wiki/H%C5%8Dk%C5%ABle%CA%BBa)
