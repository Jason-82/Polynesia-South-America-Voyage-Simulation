# Polynesia–South America Voyage Simulation: Project Spec

Oct 15, 2026 · @Jason Abrams

## Goal and success criteria

Use simulated voyages to test which travel scenario best explains the Polynesian–Native American contact around 1200 CE: Polynesians sailing to South America and back, or South Americans reaching eastern Polynesia.

Genetics can date and place the contact but can't say who traveled. A simulation can't prove that either. What it can show is which directions, routes, seasons and climate states make each scenario feasible, how long each takes, and where landfalls concentrate.

The project succeeds when all four hold:

- **Calibrated:** the model reproduces documented modern voyages and the main results of earlier simulations before any new scenario is trusted.
- **Comparative:** it reports success probabilities and durations for both directions, under drifting and under deliberate sailing, on the same footing.
- **Specific:** it names the months and climate states (El Niño, La Niña, neutral) that raise or lower the chances.
- **Useful to fieldwork:** it maps where successful landfalls concentrate on each coast, so archaeologists know where to look.

## What's known

The contact itself is settled; who traveled, how many times, and from which coast are not, and those are exactly the questions a simulation can narrow.

| Evidence | Finding | What it means for the simulation |
| --- | --- | --- |
| [Ioannidis et al. 2020](https://www.nature.com/articles/s41586-020-2487-2), modern genomes | 807 people from 17 islands and 15 Native American groups; one contact event in eastern Polynesia, before Rapa Nui was settled, with a group closest to the Zenu of present-day Colombia. The island where contact happened is unknown and may not be in the sample. | Test the Marquesas and the Colombian–Ecuadorian coast first |
| Same study, per-island admixture dates | South Marquesas (Fatu Hiva) 1150 CE; North Marquesas about 1200; Palliser and Mangareva about 1230; Rapa Nui 1380, possibly biased late by recent Chilean ancestry. Pooled estimate 1234 ± 90 CE. | South Marquesas is the earliest landing to explain |
| [Moreno-Mayar et al. 2024](https://www.nature.com/articles/s41586-024-07881-4), 15 ancient Rapanui genomes | About 10% Native American ancestry, mixed in about 1250–1430 CE; the source is reported as Andean (to confirm in the paper); where and in which direction contact happened is unresolved. An earlier study of modern Rapanui ([Moreno-Mayar et al. 2014](https://doi.org/10.1016/j.cub.2014.09.057)) allowed for one or more contacts. | Add Rapa Nui and the Peruvian coast as a second scenario |
| [Ioannidis et al. 2021](https://www.nature.com/articles/s41586-021-03902-8), settlement sequence | Society Islands about 1050 CE, Tuamotus about 1110, Mangareva about 1130, South Marquesas about 1140 (from the Tuamotus), Rapa Nui about 1200 via Mangareva. Later dates for the North Marquesas and Raivavae (1330–1360) are divergence dates, not settlement dates. | Limits which islands could launch voyages, and when |
| Radiocarbon chronologies ([Wilmshurst et al. 2011](https://doi.org/10.1073/pnas.1015876108); [Conte and Molle 2014](https://doi.org/10.1002/arco.5037)) | One settlement pulse beyond the Society Islands about 1190–1290 CE, with the Marquesas at 1200–1277 and Rapa Nui at about 1200–1253; other Marquesan excavations point earlier (Hane, Ua Huka, about 950 CE) | Dates are uncertain by decades; launch-site rules must allow for that |
| Sweet potato and bottle gourd, noted in [Ioannidis et al. 2020](https://www.nature.com/articles/s41586-020-2487-2) | American crops in the Polynesian archaeological record before Europeans arrived | A plant transfer that drift alone might explain |

**The dates are close enough to matter.** The South Marquesas admixture date (1150 CE) falls about a decade after the genetic settlement date for those islands and before the Wilmshurst radiocarbon window, though other Marquesan sites date earlier. Every one of these dates carries decades of uncertainty. So the grid runs Tuamotu and Mangareva departures alongside the Marquesas, starts Rapa Nui departures no earlier than about 1200–1250 CE, and the report says which conclusions depend on the dating.

Earlier simulations to build on:

- **1973 drift study.** Levison, Ward and Webb ran 101,016 simulated drift voyages (plus 8,052 navigated ones); with few exceptions, drifting could not explain the settlement of Polynesia from either Melanesia or South America ([summary](https://tnm.journals.yorku.ca/index.php/default/article/download/628/602)).
- **Montenegro et al. 2008.** Drift from 160 launch sites on the American coast, for sweet potato seed capsules carried by currents alone and for drifting craft pushed by the wind as well; seed capsules could reach seven island groups. As Ioannidis et al. 2020 summarize it, drift from Ecuador and Colombia is the most likely to reach Polynesia and arrives most often in the South Marquesas, then the Tuamotus. The lead author still judged Polynesians sailing to South America the likeliest route ([Nature news](https://www.nature.com/news/2007/070514/full/news070514-20.html)).
- **Montenegro, Callaghan and Fitzpatrick 2016** ([PNAS 113:12685](https://doi.org/10.1073/pnas.1612426113); [repository copy](https://repositorio.unesp.br/handle/11449/162171)). High-resolution wind and current data plus El Niño effects; success varied strongly by season and climate state, and Rapa Nui was reachable from Mangareva or the Marquesas.
- **The windward debate.** Anderson et al. argue early canoes could not sail upwind and that settlement pulses match periods of El Niño westerlies ([abstract](https://openresearch-repository.anu.edu.au/items/150a3fde-71f1-452a-835d-12b93b8f840b)); Finney's reconstructed canoes sail about 75° off the wind ([Finney](https://lir.byuh.edu/index.php/pacific/article/download/190/181/341)). A velocity prediction for double canoes by Irwin, Flay, Dudley and Johns (2023, *Archaeology in Oceania* 58:74–90) reportedly agrees on 75° and averages of up to about 4 knots; it is the best candidate source for the canoe speed table once read. The simulation should run both assumptions.
- **Making easting.** Hōkūleʻa made its 1986 eastward legs against the trades by waiting for brief westerly shifts (Finney et al. 1989, "Wait for the West Wind"), and its 1999 departure for Rapa Nui was timed to a passing low. A different route sometimes raised for reaching South America drops south out of the trades into the westerlies, runs east, then turns north along the coast; Kon-Tiki2's rafts tried it from Rapa Nui in 2016 and were abandoned about 900–1,000 nautical miles off Chile. The simulation tests both.

The gap: the research for this spec found no simulation that tests both directions against the 2020 and 2024 genetic results, including the return trip a Polynesian round voyage requires.

## Data sources

All inputs are free: modern winds and currents for the main runs, last-millennium climate runs to check that 1200 CE conditions don't change the answer, and documented voyages for calibration.

Environmental data:

| Source | What it provides | Coverage | Access |
| --- | --- | --- | --- |
| [ERA5 hourly, single levels](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels) | 10 m winds | 1940 to present; hourly; 0.25° grid | Copernicus Climate Data Store API, free account, CC-BY |
| [GLORYS12V1 ocean reanalysis](https://data.marine.copernicus.eu/product/GLOBAL_MULTIYEAR_PHY_001_030/description) | Daily surface currents (top level, 0.49 m) | 1993 to mid-2021, then the interim product to near present; 1/12° (about 8 km) | Copernicus Marine Toolbox, free account (`cmems_mod_glo_phy_my_0.083deg_P1D-m`, then `..._myint_...`) |
| NOAA [Relative Oceanic Niño Index](https://www.cpc.ncep.noaa.gov/data/indices/RONI.ascii.txt) (RONI) and [ONI](https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt) | El Niño, La Niña and neutral labels | 1950 to present, by 3-month season | Public text files |
| [HadISST Niño-3.4](https://psl.noaa.gov/data/timeseries/month/Nino34/) (NOAA PSL) | Labels for 1940–1949, before the ONI starts | 1870 to present, monthly | Public |
| [PMIP4 past1000 runs](https://gmd.copernicus.org/articles/10/4005/2017/gmd-10-4005-2017.html) | Climate-model simulations of 850–1849 CE, used as monthly anomalies | Varies by model; no model was found that published daily near-surface winds | ESGF data portals |
| [GSHHG 2.3.7](https://github.com/GenericMappingTools/gshhg-gmt/releases) | Shorelines, including small islands | Full resolution | LGPL; NOAA no longer hosts it |
| [Copernicus DEM](https://registry.opendata.aws/copernicus-dem/) GLO-30/90 | Visible heights of low islands and coastal relief | Global | Free |

Calibration and parameter sources:

| Source | What it provides | Notes |
| --- | --- | --- |
| [Hōkūleʻa's 1976 voyage](https://mauinow.com/2026/04/24/50th-anniversary-of-the-historic-launch-of-hokule%CA%BBas-1976-maiden-voyage/) | Maui (May 1) to first landfall at Mataiva in 30–31 days; Tahiti (July 4) to Oʻahu in 22 days | Finney's account says 30 days; the track is in Finney 1977 (*Science* 196:1277) and Finney 1979 |
| Hōkūleʻa's 1999 voyage to Rapa Nui | Left Mangareva September 21–22, stopped at Pitcairn, sighted Rapa Nui before dawn October 8: about 16.5 days, about 13.5 from Pitcairn | Daily navigation reports in the Polynesian Voyaging Society journals; hulls lightened and windward sails added for this leg |
| Hōkūleʻa's 1986 eastward legs | Samoa to Aitutaki and Rarotonga to Tahiti against the trades, about 9–10 days each | Finney et al. 1989; dates to confirm |
| [Hōkūleʻa specifications](https://en.wikipedia.org/wiki/H%C5%8Dk%C5%ABle%CA%BBa) | Canoe speed of 4–6 knots fully laden | — |
| Kon-Tiki, 1947 | Towed about 80–90 km off Callao on April 28; first land sighted (Puka-Puka) about day 93; wrecked on Raroia day 101 | The crew couldn't work the guaras, so the raft mostly drifted and ran before the wind |
| Kon-Tiki2, 2015–16 | Two balsa rafts: Callao (November 7, 2015) to Rapa Nui in about 42 days, with the last approach under tow; the return east via the westerlies was abandoned March 16–17, 2016 | During a strong El Niño |
| Balsa-raft engineering (Dewan and Hosler 2008, *J. Anthropological Research* 64:19–40) | Performance limits for South American sailing rafts | Not yet read; raft values stay provisional until it is |
| SAROPS leeway table (Allen and Plourde 1999; Allen 2005; distributed with [OpenDrift](https://github.com/OpenDrift/opendrift)) | Downwind and crosswind drift coefficients for 85 object classes | Small-boat, life-raft and refugee-raft classes are the nearest matches |
| Lewis, *We, the Navigators* (1972) | Land-finding ranges: terns and noddies fish within about 20 nautical miles of land, boobies 30 | Sets the bird range in the sighting rule |

Voyage figures here come partly from secondary sources (the fact-check could not open several primary pages). Before any validation check is scored, confirm its numbers from the primary account: Heyerdahl 1950 for Kon-Tiki, Finney 1979 and the Polynesian Voyaging Society logs for Hōkūleʻa, and Chilean Navy releases for Kon-Tiki2.

Notes for the builder:

- **Domain.** 30°N–50°S, 165°W–65°W: Hawaiʻi for validation, and far enough south for the westerlies route.
- **Currents before 1993.** GLORYS starts in 1993, so wind years from 1940 to 1992 pair with a monthly current climatology built from GLORYS. Flag those runs in the output. That covers all the 1947 and 1976 calibration voyages.
- **No wave fields.** Leeway coefficients are measured against the surface current and already include wave-driven drift, so adding wave (Stokes) drift would count it twice. Heavy seas enter through the wind speed at which a vessel heaves to.
- **El Niño labels.** Label with NOAA's Relative ONI, its official index since February 2026, which removes the tropical-wide warming trend; repeat with the older ONI, because the two disagree on about one season in six. For 1940–49 use HadISST Niño-3.4. Label each voyage by the 3-month season of its launch month, not the calendar year, because events straddle two years. Keep the index value as well: the record holds only a handful of very strong El Niños.
- **Early ERA5.** Before the 1950s ERA5 rests on few observations, and over the Southern Hemisphere its early record is "mainly statistical" (Soci et al. 2024, *QJRMS* 150:4014). Treat 1940s winds, including Kon-Tiki's, as the least reliable.
- **Coastlines.** Use the full-resolution GSHHG shoreline; small islands matter. Common shoreline datasets misplace some small islands by 2–3 km (Malpelo, San Félix, Cocos, Robinson Crusoe), so check each catalog island against its reference position in `configs/landmasses.yaml`.
- **Data volume.** Over the domain, 6-hourly ERA5 winds are about 1.3 GB a year, hourly about 7.5 GB, and native GLORYS surface currents about 3 GB. Regrid currents to 0.25° (a vessel covering 200 km a day gains nothing from 8 km), keep winds 6-hourly, and download hourly winds only for the stress-test years. Request ERA5 one month at a time; larger requests hit the Climate Data Store's cost limit. ARCO-ERA5 on Google Cloud needs no account, but it stores whole-globe hourly fields, so a regional extract still moves about a terabyte.

## Model design

&#91;embedded content: voyage simulator · one voyage, two decisions per step\]

Each voyage advances in 6-hour steps; thousands run in parallel, and only the statistics of their outcomes matter.

Each step, the vessel reads wind and current, picks a heading by its strategy, and moves. Then:

- **Land within sight?** If any shoreline comes inside its sighting range (set by the land's height; see below) anywhere along the step's path, the voyage ends as a landfall, recording the place and days at sea.
- **Supplies left?** This is checked only if no land is in sight. If days at sea have reached the supply limit (30, 60 or 90 days), the voyage ends as lost at sea, recording the full track. Otherwise it loops back to read conditions for the next step.

Landfall is checked first, so a boat that sights land on its last day of supplies counts as a success.

Three details of the loop matter:

- **The whole path is checked, not just where the step ends.** A canoe covers about 55 km in 6 hours, more than the range at which an atoll comes into view, so a check at step ends alone would sail past small islands.
- **Waiting in port doesn't use supplies.** Under the wait-for-wind strategy, days at sea start at departure.
- **One run serves every supply limit and sighting rule.** A voyage's track doesn't depend on either, so each voyage runs once to the longest limit and records its first sighting under every sighting rule. Success under any limit is read off afterwards: sighted at or before the limit.

What differs between voyages launched the same day: each gets its own speed factor (±10%), a steering error each step (±10°), a side for crosswind drift, and a random-walk term for currents too small for the grid to resolve. Every draw is keyed to the voyage and the step under a fixed seed, so a voyage's track never depends on which other voyages run with it, and reruns are identical.

Vessel settings (starting values, adjusted during validation; every value and its source is in `configs/vessels.yaml`):

| Vessel | Speed | Closest angle to the wind | Calms and gales | Basis |
| --- | --- | --- | --- | --- |
| Double canoe, upwind-capable | Speed table by wind angle and wind speed: about 4 knots in 10 knots of wind and 6 in 15 on a beam reach, at most 7 | About 75° | Becalmed below 2 knots of wind; heaves to and drifts from 34 knots | Hōkūleʻa; Finney; Irwin et al. 2023 |
| Double canoe, no windward sailing | Same table | 90° | Same | Tests Anderson et al.'s hypothesis |
| Sailing balsa raft | Set from Dewan and Hosler 2008; provisional until then | Same source | Same | Open question |
| Drifting craft | Current plus leeway: about 3% of wind speed downwind plus a smaller crosswind part to one side | No steering | — | SAROPS leeway table |

A sensitivity variant limits the canoe to 120° (truly downwind).

Strategy rules:

- **Drift:** no steering; the craft moves with the current and the wind's push.
- **Downwind:** sail only within the vessel's downwind range (90° or more off the wind), on the heading in that range closest to the target.
- **Hold course:** steer for a target; when it lies closer to the wind than the vessel can sail, sail close-hauled and tack when the other tack points at least 15° closer to the target.
- **Wait for wind:** stay in port until the target can be steered directly with 10° to spare in 5–25 knots of wind, then leave and hold course. Give up after 60 days in port (recorded as did not sail).
- **Latitude sailing:** reach the target's latitude, then run east or west along it, then close the target.
- **Westerlies route:** sail south into the westerlies (about 38°S), run east, then north to the target. The same rule as latitude sailing, with a different working latitude.

**Sighting land.** A landmass is in sight within the larger of two ranges:

- **Visual range:** about 3.83 × (√eye height + √land height) km, heights in metres, with standard refraction and a lookout's eye 3 m up. Land height is what shows above the horizon, including trees: about 15–20 m for Tuamotu atolls (about 21–24 km), 507 m for Rapa Nui (about 93 km), and about 960 m for Fatu Hiva (about 125 km). Coast segments use the highest relief within 5 km of the shore, from the Copernicus DEM: about 90–440 m in Colombia, 230 m in Ecuador, 100 m in northern Peru, 360–730 m in central and southern Peru, and 1,150 m in northern Chile.
- **Bird range:** terns and noddies that roost on land fish within about 20 nautical miles of it (Lewis 1972), so every island with nesting seabirds is detectable to 37 km. Coasts use the visual range only.

The original uniform rule runs as a sensitivity test at 20, 40 and 60 km (40 km is close to the bird range around an atoll), along with visual range alone and the main rule scaled by 0.5 and 1.5. Sighting another island of the home archipelago doesn't count, and the home archipelago counts again only after the voyage has been out of sight of all land.

## Experiments

Run a grid of scenarios and compare success probabilities and durations across it. Only departure, vessel and strategy need separate runs: voyages launch every day of every modern year from each departure, and month, climate state, supply limit and sighting rule are read off afterwards by filtering the same voyages.

| Dimension | Values | How it's covered |
| --- | --- | --- |
| Direction | Polynesia to South America and back; South America to Polynesia | Simulated |
| Polynesian departures | South Marquesas (Fatu Hiva), North Marquesas, Mangareva, Tuamotus; Rapa Nui only for dates after its settlement | Simulated |
| American coast segments | Colombia (7.2°N to 1.4°N), Ecuador (to 3.4°S), northern Peru (to 8°S), southern Peru (8°S to 18.35°S, including Lima), northern Chile (to 26°S); central Chile (26°S–42°S) counts as a landfall coast for the westerlies route | Departures simulated; landfall segments read off |
| Season | Every month | Read off |
| Climate state | Modern years labeled El Niño, La Niña or neutral; modern weather with last-millennium (1100–1300 CE) mean-state anomalies | Modern: read off. Last millennium: simulated |
| Vessel | Double canoe sailing 75° off the wind; double canoe sailing no closer than 90°; sailing balsa raft; unpowered drift | Simulated |
| Strategy | Drift; run downwind; hold a course; wait in port for favorable winds; latitude sailing; westerlies route (eastbound only). Drifting craft only drift. | Simulated |
| Supplies | 30, 60 or 90 days at sea; rafts also 120 (their crews could fish and catch rain) | Read off |
| Sighting rule | Height and birds (main); uniform 20, 40 or 60 km; visual only; main rule ×0.5 and ×1.5 | Read off |

Every direction runs with every vessel type, so the raft settings also cover a South American return leg.

For every cell, report:

- Success probability with a 95% interval from a bootstrap over launch years. Voyages launched in the same month share weather, so resampling individual voyages would make the intervals too narrow. Report how many distinct years each cell rests on.
- Median and 90th-percentile days at sea.
- A landfall density map for the receiving coast.
- For round trips, the joint probability of going and returning, allowing a stay of one to six months, under a return rule fixed in advance: leave on the first day after the minimum stay when winds favor the return. The best return month is also reported, marked exploratory, because picking the best of twelve noisy estimates inflates it.

The four questions the grid has to answer are fixed in advance as the primary comparisons; everything else in the grid is exploratory:

1. Can a Marquesas canoe reach the Colombian–Ecuadorian coast and get back, and in which months and climate states?
2. Can a raft from that coast, drifting or sailing, reach the South Marquesas, where the earliest admixture appears?
3. Does a Rapa Nui or Mangareva route to Peru work better for the later, Andean-linked contact?
4. How much do the answers change if canoes could not sail upwind?

Start with tests on simple made-up wind fields, then a coarse grid (one modern decade, two vessels, two strategies) to debug before scaling up.

## Validation

Trust the model on new scenarios only after it reproduces documented voyages and earlier published results. Validation runs have no supply limit (they run to 150 days), and each starts where and when the real voyage did.

| Check | Target | Tolerance |
| --- | --- | --- |
| Hōkūleʻa 1976, southbound | Maui (left May 1) to first landfall at Mataiva: 30–31 days | Duration within ±20%; landfall in the Tuamotus; track compared with Finney's published positions |
| Hōkūleʻa 1976, northbound | Tahiti (left July 4) to Oʻahu: 22 days | Duration within ±20% |
| Hōkūleʻa 1999 | Pitcairn to the first sighting of Rapa Nui (October 8): about 13.5 days, leaving on a favorable wind | Duration within ±20%; the first check with full modern data of a canoe sailing east |
| Kon-Tiki 1947 | Released about 80–90 km off Callao (April 28) to the first sighting of the Tuamotus (Puka-Puka): about 93 days, with raft settings and the guaras unused | Duration within ±25%; first sighting in the Tuamotus |
| Kon-Tiki2 2015–16 | Callao (November 7) to Rapa Nui: about 42 days; the return east from Rapa Nui (left January 6) failed | Outbound duration within ±25%; low eastbound success from Rapa Nui in that season |
| Montenegro et al. 2008 | Drift from Ecuador and Colombia arrives most often in the South Marquesas, then the Tuamotus, with that study's objects (seed capsules on currents alone; drifting craft) | Same top two island groups |
| Montenegro et al. 2016 | Rapa Nui reachable from Mangareva or the Marquesas, with strong seasonal and El Niño effects | Same qualitative pattern |

Three of the five voyage checks fall before 1993, so they use the current climatology, and Kon-Tiki falls in ERA5's least reliable years. The 1999 and 2015–16 voyages are the only checks with full modern data.

Then stress-test the results:

- **Time step:** 1 hour versus 6 hours.
- **Sighting rule:** height and birds versus uniform 20, 40 and 60 km, visual range only, and ×0.5 and ×1.5.
- **Leeway:** ±50%.
- **Canoe speed:** ±20%.
- **Random-walk dispersion:** off versus on.
- **Currents:** actual versus climatology, for 1993-onward years, to measure what the pre-1993 substitution costs.
- **Climate index:** RONI versus ONI labels.

Any headline result that flips under these changes is reported as fragile.

## Deliverables, repo layout, and acceptance criteria

The specialist hands back a tested simulator, the scenario results, and a short report with maps; one command per scenario grid reruns everything.

```
voyage-sim/
  SPEC.md
  configs/          # one YAML per scenario grid; vessels.yaml (settings and sources); landmasses.yaml
  docs/
    ARCHITECTURE.md # module contract and shared conventions
  src/voyage_sim/
    fetch.py        # ERA5 via the CDS API; GLORYS via the Copernicus Marine Toolbox; GSHHG; El Niño indices
    fields.py       # interpolate wind and current in space and time; pre-1993 current climatology
    vessels.py      # speed tables, leeway, upwind limit, calms and gales per vessel type
    strategies.py   # drift, downwind, hold course, wait for wind, latitude sailing, westerlies route
    simulate.py     # vectorized: many voyages per step; per-voyage random streams
    landfall.py     # sighting ranges by height and birds; swept-path test against shorelines
    enso.py         # El Niño, La Niña and neutral labels
    analyze.py      # probabilities, durations, year-bootstrap intervals, round-trip pairing
    plots.py        # track bundles and landfall density maps
    geo.py          # shared angle, distance and time conventions
    noise.py        # random draws keyed by voyage and step
    run.py          # one command per scenario grid
  tests/
  outputs/          # git-ignored
  REPORT.md
```

Acceptance criteria:

- [ ] All seven validation checks pass, or the report explains each miss
- [ ] Every scenario grid reruns from its config with identical results
- [ ] The report gives success probabilities and durations for both directions, by vessel, strategy, month and climate state, with year-bootstrap intervals and the number of years behind each
- [ ] Landfall density maps exist for both coasts
- [ ] A parameter table lists every vessel and strategy setting with its source
- [ ] Sensitivity results cover time step, sighting rule, leeway, speed, dispersion, currents and climate index

Check with the project owner before: buying cloud compute, contacting researchers, or publishing anything.

## Milestones, risks, and open questions

A validated simulator is about a week of part-time work; the full grid and report take two to three weeks more.

1. **Core (days 1–3):** data access, wind and current interpolation, vessel and strategy rules, the landfall test, and the integrator, all tested on simple made-up wind fields.
2. **Validation (days 4–6):** confirm the voyage figures from primary accounts, then the seven checks and the stress tests.
3. **Grid (week 2):** the full scenario grid on modern climate.
4. **Past climate (week 3):** rerun the key cells with last-millennium anomalies added to modern weather.
5. **Report (weeks 3–4):** probability tables, maps, and a plain-language summary.

| Risk | Early signal | Fallback |
| --- | --- | --- |
| Modern winds differ from 1200 CE climate | Last-millennium runs disagree with modern results | Report both; treat agreement as robust and disagreement as uncertain |
| Climate models are biased in the tropical Pacific | A model's historical run misses observed trade winds by more than the 1200 CE signal | Use each model's past1000-minus-historical anomaly added to ERA5, never raw model winds; report the spread across models |
| Vessel performance is uncertain | Validation durations miss their tolerance | Calibrate speed tables to Hōkūleʻa; report results as ranges |
| The sighting rule drives the answer | Probabilities swing widely across sighting rules | Report the spread rather than one number |
| Human strategy is hard to model | Results depend mostly on the strategy chosen | Bracket with simple strategies; don't claim a single best route |
| Few strong El Niño years | El Niño cells rest on a handful of years | Report the number of years; use the index value as a continuous covariate; treat the split as uncertain |
| Data download is slow or large | The Climate Data Store queue or disk space stalls the first decade | Regrid currents to 0.25°, keep winds 6-hourly, fetch hourly winds only for stress-test years |
| Calibration figures are wrong | A primary account disagrees with the numbers above | Correct the target before scoring; never tune the model to an unconfirmed number |

Open questions for the owner:

- [ ] Which raft performance numbers to adopt from Dewan and Hosler 2008? Until the article is read, raft runs use provisional values bracketed by Kon-Tiki and Kon-Tiki2.
- [ ] Run Marquesas departures for the earliest contact, given that the dates are uncertain by decades? (Recommended: yes, reported alongside the Tuamotus and Mangareva.)
- [ ] Ask a Pacific voyaging researcher to review the strategy rules before the full grid, and ask the Polynesian Voyaging Society for logs or tracker positions to fit the canoe speed table?
- [ ] Publish the results, or keep them as a personal study?

Sources: [Ioannidis et al. 2020](https://www.nature.com/articles/s41586-020-2487-2) · [Ioannidis et al. 2021](https://www.nature.com/articles/s41586-021-03902-8) · [Moreno-Mayar et al. 2024](https://www.nature.com/articles/s41586-024-07881-4) · [Moreno-Mayar et al. 2014](https://doi.org/10.1016/j.cub.2014.09.057) · [Wilmshurst et al. 2011](https://doi.org/10.1073/pnas.1015876108) · [Conte and Molle 2014](https://doi.org/10.1002/arco.5037) · [Montenegro et al. 2016](https://doi.org/10.1073/pnas.1612426113) · [Nature news on Montenegro et al. 2008](https://www.nature.com/news/2007/070514/full/news070514-20.html) · [Computer simulations of ancient voyaging](https://tnm.journals.yorku.ca/index.php/default/article/download/628/602) · [Anderson et al. 2006](https://openresearch-repository.anu.edu.au/items/150a3fde-71f1-452a-835d-12b93b8f840b) · [Finney on canoe performance](https://lir.byuh.edu/index.php/pacific/article/download/190/181/341) · Irwin et al. 2023, *Archaeology in Oceania* 58:74–90 · Finney et al. 1989, *J. Polynesian Society* 98:261–302 · Dewan and Hosler 2008, *J. Anthropological Research* 64:19–40 · Lewis 1972, *We, the Navigators* · Allen and Plourde 1999, USCG CG-D-08-99 · Soci et al. 2024, *QJRMS* 150:4014–4048 · [ERA5](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels) · [GLORYS12V1](https://data.marine.copernicus.eu/product/GLOBAL_MULTIYEAR_PHY_001_030/description) · [NOAA RONI](https://www.cpc.ncep.noaa.gov/data/indices/RONI.ascii.txt) · [PMIP4 past1000](https://gmd.copernicus.org/articles/10/4005/2017/gmd-10-4005-2017.html) · [GSHHG](https://github.com/GenericMappingTools/gshhg-gmt/releases) · [Hōkūleʻa 1976 timeline](https://mauinow.com/2026/04/24/50th-anniversary-of-the-historic-launch-of-hokule%CA%BBas-1976-maiden-voyage/) · [Hōkūleʻa specifications](https://en.wikipedia.org/wiki/H%C5%8Dk%C5%ABle%CA%BBa)
