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
| Radiocarbon chronologies ([Wilmshurst et al. 2011](https://doi.org/10.1073/pnas.1015876108); [Conte and Molle 2014](https://doi.org/10.1002/arco.5037)) | One settlement pulse beyond the Society Islands about 1190–1290 CE, with the Marquesas at 1200–1277 and Rapa Nui at about 1200–1253; other Marquesan excavations point earlier (Hane, Ua Huka, about 950 CE) | Dates are uncertain by decades; the report says which conclusions depend on them |
| Sweet potato and bottle gourd, noted in [Ioannidis et al. 2020](https://www.nature.com/articles/s41586-020-2487-2) | American crops in the Polynesian archaeological record before Europeans arrived | A plant transfer that drift alone might explain |

**The dates are close enough to matter.** The South Marquesas admixture date (1150 CE) falls about a decade after the genetic settlement date for those islands and before the Wilmshurst radiocarbon window, though other Marquesan sites date earlier. Every one of these dates carries decades of uncertainty. So the grid runs Tuamotu and Mangareva departures alongside the Marquesas, uses Rapa Nui departures only for the later contact, and the report says which conclusions depend on the dating.

Earlier simulations to build on:

- **1973 drift study.** Levison, Ward and Webb ran 101,016 simulated drift voyages (plus 8,052 navigated ones); with few exceptions, drifting could not explain the settlement of Polynesia from either Melanesia or South America ([summary](https://tnm.journals.yorku.ca/index.php/default/article/download/628/602)).
- **Montenegro et al. 2008.** Drift from 160 launch sites on the American coast, for sweet potato seed capsules carried by currents alone and for drifting craft pushed by the wind as well; seed capsules could reach seven island groups. As Ioannidis et al. 2020 summarize it, drift from Ecuador and Colombia is the most likely to reach Polynesia and arrives most often in the South Marquesas, then the Tuamotus. The lead author still judged Polynesians sailing to South America the likeliest route ([Nature news](https://www.nature.com/news/2007/070514/full/news070514-20.html)). The site count, island groups and drifted objects come from the news story; confirm them from the paper (*J. Archaeological Science* 35:355–367).
- **Montenegro, Callaghan and Fitzpatrick 2016** ([PNAS 113:12685](https://doi.org/10.1073/pnas.1612426113); [possible repository copy, unverified](https://repositorio.unesp.br/handle/11449/162171)). High-resolution wind and current data plus El Niño effects; success varied strongly by season and climate state, and Rapa Nui was reachable from Mangareva or the Marquesas (to re-check against the paper).
- **The windward debate.** Anderson et al. argue early canoes could not sail upwind and that settlement pulses match periods of El Niño westerlies ([abstract](https://openresearch-repository.anu.edu.au/items/150a3fde-71f1-452a-835d-12b93b8f840b)); Finney's reconstructed canoes sail about 75° off the wind ([Finney](https://lir.byuh.edu/index.php/pacific/article/download/190/181/341)). A velocity prediction for double canoes by Irwin, Flay, Dudley and Johns (2023, *Archaeology in Oceania* 58:74–90) reportedly agrees on 75° and averages of up to about 4 knots; once read, it is the best source for the canoe speed table. The simulation should run both assumptions.
- **Sailing east against the trades.** Hōkūleʻa made its 1986 eastward legs by waiting for brief westerly shifts (Finney et al. 1989, "Wait for the West Wind"), and its 1999 departure for Rapa Nui was timed to a passing low. A different route sometimes raised for reaching South America drops south out of the trades into the westerlies, runs east, then turns north along the coast; Kon-Tiki2's rafts tried it from Rapa Nui in 2016 and were abandoned about 900–1,000 nautical miles off Chile. The simulation tests both, as the wait-for-wind and westerlies-route strategies.

The gap: the research for this spec found no simulation that tests both directions against the 2020 and 2024 genetic results, including the return trip a Polynesian round voyage requires. The literature search was cut short, though, and two leads are unread: Fitzpatrick and Callaghan 2009 (*J. Archaeological Science* 36:214–223), a sailing simulation from Polynesia to South America, and a 2022 *Quaternary Science Reviews* study of sweet-potato drift.

## Data sources

All inputs are free: modern winds and currents for the main runs, last-millennium climate runs to check that 1200 CE conditions don't change the answer, and documented voyages for calibration.

Environmental data:

| Source | What it provides | Coverage | Access |
| --- | --- | --- | --- |
| [ERA5 hourly, single levels](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels) | 10 m winds | 1940 to present; hourly; 0.25° grid | Copernicus Climate Data Store API, free account, CC-BY |
| [GLORYS12V1 ocean reanalysis](https://data.marine.copernicus.eu/product/GLOBAL_MULTIYEAR_PHY_001_030/description) | Daily surface currents (top level, 0.49 m) | `cmems_mod_glo_phy_my_0.083deg_P1D-m` from 1993 to June 30, 2021, then `cmems_mod_glo_phy_myint_0.083deg_P1D-m`; 1/12° (about 8 km) | Copernicus Marine Toolbox, free account |
| NOAA [Relative Oceanic Niño Index](https://www.cpc.ncep.noaa.gov/data/indices/RONI.ascii.txt) (RONI) and [ONI](https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt) | El Niño, La Niña and neutral labels. RONI, NOAA's official index since February 2026, subtracts the tropical-wide mean temperature | 1950 to present, by 3-month season | Public text files |
| [HadISST Niño-3.4](https://psl.noaa.gov/data/timeseries/month/Nino34/) (NOAA PSL) | Labels for 1940–1949, before the ONI starts | 1870 to present, monthly | Public |
| [PMIP4 past1000 runs](https://gmd.copernicus.org/articles/10/4005/2017/gmd-10-4005-2017.html) | Monthly winds from the past1000 (850–1849 CE) and historical (1850–2014) runs, used as anomalies | Varies by model; no past1000 model has yet been found with daily wind components (MRI-ESM2-0 has daily wind speed only); the ESGF search is still to do | ESGF data portals |
| [GSHHG 2.3.7](https://github.com/GenericMappingTools/gshhg-gmt/releases) | Shorelines, including small islands | Full resolution | LGPL; NOAA no longer hosts it |
| [Copernicus DEM](https://registry.opendata.aws/copernicus-dem/) (digital elevation model; GLO-90 is its 90 m grid) | Coastal relief for the sighting rule; heights of low islands including trees | Global | Free |

Calibration and parameter sources:

| Source | What it provides | Notes |
| --- | --- | --- |
| [Hōkūleʻa's 1976 voyage](https://mauinow.com/2026/04/24/50th-anniversary-of-the-historic-launch-of-hokule%CA%BBas-1976-maiden-voyage/) | Maui (May 1) to first landfall at Mataiva in 30–31 days; Tahiti (July 4) to Oʻahu in 22 days | Finney's account says 30 days. Track maps are reportedly in the Polynesian Voyaging Society archive and Finney 1979; no daily-position table has been located yet |
| Hōkūleʻa's 1999 voyage to Rapa Nui | Left Mangareva under tow on September 21 or 22, stopped at Pitcairn and left late September 24, sighted Rapa Nui before dawn October 8: about 15.5–16.5 days, about 13.3 from Pitcairn | Daily navigation reports in the Polynesian Voyaging Society journals; hulls lightened and windward sails added for this leg |
| Hōkūleʻa's 1986 eastward legs | Samoa to Aitutaki and Rarotonga to Tahiti against the trades, about 9–10 days each | Finney et al. 1989. Context for the wait-for-wind rule, not a check (Samoa lies outside the domain) |
| [Hōkūleʻa specifications](https://en.wikipedia.org/wiki/H%C5%8Dk%C5%ABle%CA%BBa) | Canoe speed of 4–6 knots fully laden | — |
| Kon-Tiki, 1947 | Towed 50 miles off Callao (80–93 km depending on the unit; Ioannidis et al. 2020 say 80 miles) and released April 28 or 29; Puka-Puka sighted July 30, day 93 from departure; wrecked on Raroia August 7, day 101 | The crew couldn't work the guaras (removable centreboards used to steer and hold a course off the wind), so the raft mostly drifted and ran before the wind |
| Kon-Tiki2, 2015–16 | Two balsa rafts left Callao November 7, 2015; one moored off Rapa Nui December 19 (day 42) after a towed approach. The return east via the westerlies, from January 6, was abandoned March 16–17 about 900–1,000 nautical miles off Chile | During a strong El Niño. The first-sighting date of Rapa Nui is still to confirm from the log |
| Balsa-raft engineering (Dewan and Hosler 2008, *J. Anthropological Research* 64:19–40) | Performance limits for South American sailing rafts | Not yet read; raft values stay provisional until it is |
| SAROPS (US Coast Guard Search and Rescue Optimal Planning System) leeway table (Allen and Plourde 1999; Allen 2005; distributed with [OpenDrift](https://github.com/OpenDrift/opendrift)) | Downwind and crosswind drift coefficients for 85 object classes | Canoes use class SKIFF-1 and rafts REFUGEE-RAFT-1; the dismasted-sailboat classes are an alternative for a disabled canoe |
| Lewis, *We, the Navigators* (1972) | Land-finding ranges: terns and noddies fish within about 20 miles of land, boobies 30 | Quoted second-hand; page and unit to confirm (37 km if nautical, 32 km if statute) |

The voyage figures and the Lewis bird ranges come partly from secondary sources, because the fact-check could not open several primary pages. Before any validation check is scored, confirm its numbers from the primary account: Heyerdahl 1950 for Kon-Tiki, Finney 1979 and the Polynesian Voyaging Society logs for Hōkūleʻa, and Chilean Navy releases for Kon-Tiki2. Citation details for Irwin et al. 2023, Finney et al. 1989, Dewan and Hosler 2008 and Allen and Plourde 1999 were seen only in search excerpts; confirm them before the spec is shared.

Notes for the builder:

- **Domain.** 30°N–50°S, 165°W–65°W: Hawaiʻi for validation, and far enough south for the westerlies route.
- **Currents before 1993.** GLORYS starts in 1993, so wind years from 1940 to 1992 pair with a monthly current climatology built from GLORYS. Flag those runs in the output. That covers all the 1947 and 1976 calibration voyages.
- **No wave fields.** Leeway coefficients are measured against the surface current and already include wave-driven drift, so adding wave (Stokes) drift would count it twice. Heavy seas enter through the wind speed at which a vessel heaves to.
- **El Niño labels.** Label each voyage by the 3-month season centred on its launch month (an April launch takes MAM). Use RONI: El Niño at +0.5 °C or more and La Niña at −0.5 °C or less, held for at least five overlapping seasons; otherwise neutral. Repeat with the older ONI, because the two disagree on about one season in six. For 1940–49, re-centre the HadISST Niño-3.4 anomaly on 1991–2020, apply the same rule, and flag those labels as a different index. Keep the index value too: the record holds only a handful of very strong El Niños.
- **Early ERA5.** Before the 1950s ERA5 rests on few observations, and over the Southern Hemisphere its early record is "mainly statistical" (Soci et al. 2024, *QJRMS* 150:4014). Keep 1940–49 in the grid, flag those voyages, and report each headline result with and without them. Cap 10 m winds at 50 m/s and log every capped value: ERA5 occasionally analyses spurious winds of up to about 300 m/s.
- **Distance to land.** For each sighting rule, precompute once a grid of sighting margin at 0.05°: distance to the nearest shoreline minus that landmass's sighting range. The whole-path landfall test samples this grid along each step.
- **Coastlines.** Use the full-resolution GSHHG shoreline; small islands matter. Natural Earth misplaces several small islands (Sala y Gómez by about 10 km; Malpelo, San Félix, Cocos and Robinson Crusoe by 2–3 km), and GSHHG has not been checked yet, so compare each catalog island with its reference position in `configs/landmasses.yaml`, Sala y Gómez first, and move its shoreline when they differ by more than 1 km.
- **Data volume.** Keep winds 6-hourly (about 1.4 GiB a year over the domain) and regrid currents to 0.25°; a canoe covering 200 km a day gains nothing from 8 km. Download hourly winds only for the 1-hour time-step test on the coarse-grid decade. Request ERA5 one month at a time to stay under the Climate Data Store's request limit.

## Model design

&#91;embedded content: voyage simulator · one voyage, two decisions per step\]

Each voyage advances in 6-hour steps; thousands run in parallel, and only the statistics of their outcomes matter.

Each step, the vessel reads wind and current, picks a heading by its strategy, and moves. Then:

- **Land within sight?** If a destination shoreline comes inside its sighting range (set by the land's height; see below) anywhere along the step's path, the voyage ends as a landfall, recording the place and days at sea.
- **Supplies left?** This is checked only if no land is in sight. If days at sea have reached the supply limit (30, 60 or 90 days; rafts also 120), the voyage ends as lost at sea, recording the full track. Otherwise it loops back to read conditions for the next step.

Landfall is checked first, so a boat that sights land on its last day of supplies counts as a success.

The loop describes one voyage under one sighting rule and one supply limit. Five details matter:

- **Only a destination ends a voyage.** For a voyage leaving Polynesia, the destination is the American coast from Colombia to central Chile; for a voyage leaving the Americas, any island of eastern Polynesia (the Marquesas, Tuamotus, Society, Austral and Gambier Islands, the Pitcairn group and Rapa Nui); a round trip's return leg aims for home. Other land sighted on the way, such as Pitcairn for an eastbound canoe or the Galápagos, Cocos, Malpelo, the Desventuradas and Juan Fernández, is recorded and reported separately, but the voyage sails on. A return leg counts home only after the voyage has been out of sight of all land.
- **The whole path is checked, not just where the step ends.** A canoe covers about 55 km in 6 hours, more than the range at which an atoll comes into view, so a check at step ends alone would sail past small islands.
- **Waiting in port doesn't use supplies.** Under the wait-for-wind strategy, days at sea start at departure. Month and climate state are those of the launch day, when waiting starts. A voyage that never sails counts as a failure, and each cell also reports the share that never sailed.
- **Leaving the domain is lost at sea.**
- **One run serves every supply limit and sighting rule.** A voyage's track doesn't depend on either, so in the code a voyage keeps sailing after a sighting until every sighting rule has sighted its destination, or until the longest limit (90 days; 120 for rafts; 150 in validation runs). The first sighting under each rule is recorded, and success under any limit is read off afterwards: sighted at or before the limit.

What differs between voyages launched the same day: a speed factor drawn once from a normal distribution (mean 1, standard deviation 0.10, floored at 0.3); a steering error each step (normal, standard deviation 10°); a side for crosswind drift; and a random walk standing in for currents too small for the grid to resolve, with diffusivity 30 m²/s for 0.25° currents (published practice spans about 10–100 m²/s). Every draw is keyed to the voyage and the step under a fixed seed, so a voyage's track never depends on which other voyages run with it, and reruns are identical.

Vessel settings (starting values, adjusted during validation; every value and its source is in `configs/vessels.yaml`):

| Vessel | Speed | Closest angle to the wind | Calms and gales | Basis |
| --- | --- | --- | --- | --- |
| Double canoe, upwind-capable | Speed table by wind angle and wind speed: about 4 knots in 10 knots of wind and 6 in 15 on a beam reach (wind at 90°), at most 7 | About 75° | Becalmed below 2 knots of wind; heaves to and drifts from 34 knots | Hōkūleʻa specifications and Finney, provisional; to be replaced by Irwin et al. 2023 once read, which reports lower speeds at 75° in strong wind |
| Double canoe, no windward sailing | Same table | 90° | Same | Tests Anderson et al.'s hypothesis |
| Sailing balsa raft | Provisional: at most 4 knots | 85°, provisional | Becalmed below 3 knots; heaves to from 34 knots | Dewan and Hosler 2008, once read |
| Drifting canoe | Current plus leeway: 3.15% of wind speed downwind, 1.29% crosswind | No steering | — | SAROPS class SKIFF-1 |
| Drifting raft | Current plus leeway: 1.56% of wind speed plus 8.3 cm/s downwind, a little crosswind | No steering | — | SAROPS class REFUGEE-RAFT-1 |

A sailing vessel that is becalmed or hove to drifts with its class's leeway. Validation adds two settings: a seed capsule that moves with the current alone (Montenegro et al. 2008), and the sailing raft with its guaras unused, which can't sail closer than 120° (Kon-Tiki). A sensitivity variant limits the no-windward canoe to 120° off the wind (broad reach and run only).

Strategy rules:

- **Drift:** no steering; the craft moves with the current and the wind's push. Drifting craft only drift, and sailing vessels never use this rule.
- **Downwind:** sail only within the vessel's downwind range (90° or more off the wind), on the heading in that range closest to the target.
- **Hold course:** steer for a target; when it lies closer to the wind than the vessel can sail, sail close-hauled (at the vessel's closest angle to the wind) and tack when the other tack points at least 15° closer to the target.
- **Wait for wind:** stay in port until the target can be steered directly with 10° to spare in 5–25 knots of wind, then leave and hold course. Give up after 60 days in port (recorded as did not sail).
- **Latitude sailing:** reach the target's latitude, then run east or west along it, then close the target.
- **Westerlies route (eastbound only):** sail south into the westerlies (about 38°S), run east, then north to the target. The same rule as latitude sailing, with a different working latitude.

**Targets.** Every rule but drift steers for a target. Proposed, for the owner to confirm: Marquesas and Tuamotu departures aim for Manta, Ecuador (question 1); Mangareva and Rapa Nui departures aim for Callao, Peru (question 3); American departures aim for Fatu Hiva (question 2) and Rapa Nui (question 3).

**Sighting land.** A landmass is in sight within the larger of two ranges:

- **Visual range:** 3.83 × (√e + √h) km with standard refraction, where e = 3 m (a lookout's eye) and h is the land's highest point in metres, counting the trees on atolls. Tuamotu atolls are 15–20 m (21–24 km), Rapa Nui 507 m (93 km), and Fatu Hiva about 960 m (125 km; literature values run to 1,125 m, about 135 km).
- **Bird range:** 37 km around every island, assuming all had nesting seabirds: terns and noddies fish within about 20 nautical miles of land (Lewis 1972; unit to confirm). Coasts use the visual range only.

Coasts are split into 0.1° latitude pieces, and each piece's h is the highest point within 5 km of its shore in the Copernicus DEM. Typical values: 440 m on the Baudó coast of Colombia (7.2–4.5°N) and 90 m south of it, 230 m in Ecuador, 100 m in northern Peru, 360 m from 8°S to 14°S and 730 m from 14°S to 18.35°S, and 1,150 m in northern Chile. Central Chile is still to compute.

Sensitivity tests use a uniform 20, 40 or 60 km for all land (the original spec's rule; 40 km is close to the bird range around an atoll), visual range alone, and the main rule with both ranges scaled by 0.5 and 1.5.

## Experiments

Run a grid of scenarios and compare success probabilities and durations across it. Departure, target, vessel, strategy and climate background (modern, or modern plus last-millennium anomalies) need separate runs, as do the stress tests that change the physics. Month, modern climate state, supply limit and sighting rule are read off afterwards by filtering the same voyages. From each departure point, 4 voyages launch at 00 UTC every day of every year from 1940 to 2025.

| Dimension | Values | How it's covered |
| --- | --- | --- |
| Direction | Polynesia to South America and back; South America to Polynesia | Simulated |
| Polynesian departures | Fatu Hiva (South Marquesas), Nuku Hiva (North Marquesas), Mangareva, Pukarua (eastern Tuamotus), Rapa Nui. The weather is modern, so every departure runs on every year; settlement dates decide only which question a departure answers (Rapa Nui answers question 3, never the 1150 CE contact) | Simulated |
| American departures | One port per coast segment: Tumaco (Colombia, 7.2°N to 1.4°N), Manta (Ecuador, to 3.4°S), Paita (northern Peru, to 8°S), Callao (southern Peru, 8°S to 18.35°S), Arica (northern Chile, to 26°S). Drift runs also launch every 100 km along the coast | Simulated |
| Targets | As proposed under Model design | Simulated |
| Destination coasts | The five segments plus central Chile (26°S–42°S); offshore islands reported separately | Read off |
| Season | Every month | Read off |
| Climate state | Modern launches labeled El Niño, La Niña or neutral by their launch season. Last millennium: for each PMIP4 model, the 12 calendar-month mean winds of its past1000 run over 1100–1300 CE, minus the same from its historical run over 1940–2014, added to every ERA5 wind field; currents unchanged and flagged; one set per model | Modern: read off. Last millennium: simulated |
| Vessel | Double canoe sailing 75° off the wind; double canoe sailing no closer than 90°; sailing balsa raft; drifting canoe; drifting raft | Simulated |
| Strategy | Sailing vessels: run downwind; hold a course; wait in port for favorable winds; latitude sailing; westerlies route (eastbound only). Drifting craft: drift | Simulated |
| Supplies | 30, 60 or 90 days at sea; rafts, sailing or drifting, also 120 (their crews could fish and catch rain) | Read off |
| Sighting rule | Height and birds (main); uniform 20, 40 or 60 km; visual only; main rule ×0.5 and ×1.5 | Read off |

Every direction runs with every vessel type. A raft run from Polynesia to South America is the return leg South Americans who reached Polynesia would have sailed, so no separate South American round trip is needed. Question 2's drifting case uses the drifting raft.

For every cell, report:

- Success probability with a 95% interval from a bootstrap over July-to-June years, so one El Niño or La Niña event is one cluster. Voyages in the same season share weather, so resampling individual voyages would make the intervals too narrow. Report how many distinct years each cell rests on, and for El Niño and La Niña cells how many distinct events.
- Median and 90th-percentile days at sea.
- A landfall density map for the receiving coast, plotting the shoreline point nearest each first sighting of the destination.
- For round trips, the joint probability of going and returning, under a return rule fixed in advance. The return leg starts at the shoreline point nearest where the outbound voyage first sighted land. For a minimum stay of M months (M = 1 to 6, each reported), it leaves on the first day at least M months after that sighting on which the wait-for-wind test passes for the home island; if no such day comes within 60 days, the round trip fails. It uses the same vessel, the hold-course rule and the same supply limit, and it succeeds on sighting the home archipelago. Joint probability is the share of all outbound launches that complete both legs. The best return month is also reported, marked exploratory, because picking the best of twelve noisy estimates inflates it.

The four questions the grid has to answer are fixed in advance as the primary comparisons; everything else in the grid is exploratory:

1. Can a Marquesas canoe reach the Colombian–Ecuadorian coast and get back, and in which months and climate states?
2. Can a raft from that coast, drifting or sailing, reach the South Marquesas, where the earliest admixture appears?
3. Does a Rapa Nui or Mangareva route to Peru work better for the later, Andean-linked contact?
4. How much do the answers change if canoes could not sail upwind?

Start with tests on simple made-up wind fields, then a coarse grid (2010–2019, two vessels, two strategies) to debug before scaling up.

## Validation

Trust the model on new scenarios only after it reproduces documented voyages and earlier published results. Validation runs have no supply limit (they run to 150 days). Each voyage check launches 1,000 voyages from the real start point and time and compares the median days to the first sighting of the destination; a landfall check passes when that destination is the most common first sighting. Tracks are compared in the report but not scored. If the raft settings are adjusted to match Kon-Tiki or Kon-Tiki2, the report calls those two checks calibration, not validation.

| Check | Setup | Target | Tolerance |
| --- | --- | --- | --- |
| Hōkūleʻa 1976, southbound | Upwind canoe, hold course for Tahiti, from Honolua Bay, Maui, May 1 | First landfall at Mataiva: 30–31 days | Duration within ±20%; landfall in the Tuamotus |
| Hōkūleʻa 1976, northbound | Upwind canoe, hold course for Oʻahu, from Papeete, July 4 | 22 days to Oʻahu, less the time from first sighting to arrival (from the log) | Duration within ±20% |
| Hōkūleʻa 1999 | Upwind canoe with unchanged settings, hold course for Rapa Nui, from Pitcairn late on September 24 | First sighting of Rapa Nui before dawn October 8: about 13.3 days | Duration within ±20%. The real canoe was lightened for this leg, so a miss is reported rather than the canoe retuned |
| Kon-Tiki 1947 | Sailing raft with its guaras unused, downwind toward Raroia, released 50 miles off Callao | First sighting of the Tuamotus (Puka-Puka): about 93 days from departure | Duration within ±25%; first sighting in the Tuamotus |
| Kon-Tiki2 2015–16 | Sailing raft, hold course for Rapa Nui from Callao, November 7; then the westerlies route from Rapa Nui toward Chile, January 6 | Outbound: first sighting of Rapa Nui, before day 42 (date to confirm). Return: the real attempt failed after 70 days | Outbound duration within ±25%; return: fewer than a quarter of runs reach the American coast within 70 days |
| Montenegro et al. 2008 | Drifting canoe and seed capsule, launched daily from points along Ecuador and Colombia in 1993-onward years | Drifting craft arrive most often in the South Marquesas, then the Tuamotus (as Ioannidis et al. 2020 report it); the seed-capsule ranking is reported, not scored | Same top two island groups |
| Montenegro et al. 2016 | Upwind canoe, hold course for Rapa Nui from Mangareva and the Marquesas, launched daily; and drift from Mangareva in May–July | Rapa Nui reachable by sailing, with strong seasonal and El Niño effects (to re-check against the paper); drift almost never arrives (Levison et al. 1973 report none of 2,208 May–July drifts) | Same qualitative pattern |

Three of the five voyage checks fall before 1993, so they use the current climatology, and Kon-Tiki falls in ERA5's least reliable years. The 1999 and 2015–16 voyages are the only voyage checks with full modern data; the two Montenegro checks run on 1993-onward years.

Then stress-test the results:

- **Time step:** 1 hour versus 6 hours.
- **Sighting rule:** height and birds versus uniform 20, 40 and 60 km, visual range only, and ×0.5 and ×1.5.
- **Leeway:** ±50%.
- **Canoe speed:** ±20%.
- **Windward limit:** the no-windward canoe at 90° versus 120°.
- **Random walk:** diffusivity 30 m²/s versus 0 and 100 m²/s.
- **Currents:** actual versus climatology, for 1993-onward years, to measure what the pre-1993 substitution costs.
- **Climate index:** RONI versus ONI labels.
- **Drift integrator:** one year of drifting-canoe launches from Colombia and Ecuador through OpenDrift's Leeway model on the same fields, with the random walk off, comparing arrival island groups and median days.

Any headline result that flips under these changes is reported as fragile.

## Deliverables, repo layout, and acceptance criteria

The specialist hands back a tested simulator, the scenario results, and a short report with maps; one command per scenario grid reruns everything.

```
voyage-sim/
  SPEC.md
  configs/          # one YAML per scenario grid; vessels.yaml (settings and sources);
                    # landmasses.yaml (island groups, heights, reference positions)
  docs/
    ARCHITECTURE.md # module contract and shared conventions
  src/voyage_sim/
    fetch.py        # ERA5 via the CDS API; GLORYS via the Copernicus Marine Toolbox; GSHHG;
                    # El Niño indices; PMIP4 monthly winds via ESGF
    fields.py       # interpolate wind and current in space and time; pre-1993 current
                    # climatology; last-millennium wind anomalies
    vessels.py      # speed tables, leeway, upwind limit, calms and gales per vessel type
    strategies.py   # drift, downwind, hold course, wait for wind, latitude sailing, westerlies route
    catalog.py      # landmasses from GSHHG shorelines, with groups and sighting heights
    landfall.py     # sighting ranges by height and birds; sighting-margin grid; whole-path test
    simulate.py     # vectorized: many voyages per step; per-voyage random streams
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
- [ ] Round-trip joint probabilities under the fixed return rule exist for every Polynesian departure, with the best return month marked exploratory
- [ ] Landfall density maps exist for both coasts
- [ ] The cells behind the four primary questions are rerun with last-millennium anomalies and compared with modern results
- [ ] The report answers the four primary questions explicitly
- [ ] A parameter table lists every vessel and strategy setting with its source
- [ ] Sensitivity results cover time step, sighting rule, leeway, speed, windward limit, random walk, currents, climate index and the OpenDrift cross-check

Check with the project owner before: buying cloud compute, contacting researchers, or publishing anything.

## Milestones, risks, and open questions

A validated simulator is about a week of part-time work; the full grid and report take two to three weeks more.

1. **Core (days 1–3):** data access, wind and current interpolation, vessel and strategy rules, the landfall test, and the integrator, all tested on simple made-up wind fields.
2. **Validation (days 4–6):** confirm the voyage figures from primary accounts, then the seven checks, with the time-step test on the validation voyages.
3. **Grid (week 2):** the full scenario grid on modern climate, then the stress tests on its headline results.
4. **Past climate (week 3):** rerun the cells behind the four primary questions with last-millennium anomalies added to modern weather.
5. **Report (weeks 3–4):** probability tables, maps, and a plain-language summary.

| Risk | Early signal | Fallback |
| --- | --- | --- |
| Modern winds differ from 1200 CE climate | Last-millennium runs disagree with modern results | Report both; treat agreement as robust and disagreement as uncertain |
| Climate models are biased in the tropical Pacific | A model's historical run misses observed trade winds by more than the 1200 CE signal | Use model anomalies added to ERA5, never raw model winds; report the spread across models |
| Vessel performance is uncertain | Validation durations miss their tolerance | Calibrate speed tables to Hōkūleʻa; report results as ranges |
| The sighting rule drives the answer | Probabilities swing widely across sighting rules | Report the spread rather than one number |
| Human strategy is hard to model | Results depend mostly on the strategy chosen | Bracket with simple strategies; don't claim a single best route |
| Few strong El Niño years | El Niño cells rest on a handful of events | Report the number of events; use the index value as a continuous covariate; treat the split as uncertain |
| Data download is slow | The Climate Data Store queue stalls the first decade | Start the grid on the years already downloaded and fetch the rest in the background |
| Calibration figures are wrong | A primary account disagrees with the numbers above | Correct the target before scoring; never tune the model to an unconfirmed number |

Open questions for the owner:

- [ ] Which raft performance numbers to adopt from Dewan and Hosler 2008? Until the article is read, raft runs use the provisional values in `configs/vessels.yaml` (closest angle 85°, at most 4 knots).
- [ ] Confirm the departure points and steering targets proposed above, especially the Tuamotu atoll: Pukarua in the east, or the Palliser group, which has an admixture date.
- [ ] Ask a Pacific voyaging researcher to review the strategy rules before the full grid, and ask the Polynesian Voyaging Society for logs or tracker positions to fit the canoe speed table?
- [ ] Publish the results, or keep them as a personal study?

Sources: [Ioannidis et al. 2020](https://www.nature.com/articles/s41586-020-2487-2) · [Ioannidis et al. 2021](https://www.nature.com/articles/s41586-021-03902-8) · [Moreno-Mayar et al. 2024](https://www.nature.com/articles/s41586-024-07881-4) · [Moreno-Mayar et al. 2014](https://doi.org/10.1016/j.cub.2014.09.057) · [Wilmshurst et al. 2011](https://doi.org/10.1073/pnas.1015876108) · [Conte and Molle 2014](https://doi.org/10.1002/arco.5037) · [Montenegro et al. 2016](https://doi.org/10.1073/pnas.1612426113) · [Nature news on Montenegro et al. 2008](https://www.nature.com/news/2007/070514/full/news070514-20.html) · [Computer simulations of ancient voyaging](https://tnm.journals.yorku.ca/index.php/default/article/download/628/602) · [Anderson et al. 2006](https://openresearch-repository.anu.edu.au/items/150a3fde-71f1-452a-835d-12b93b8f840b) · [Finney on canoe performance](https://lir.byuh.edu/index.php/pacific/article/download/190/181/341) · Irwin et al. 2023, *Archaeology in Oceania* 58:74–90 · Finney et al. 1989, *J. Polynesian Society* 98:261–302 · Dewan and Hosler 2008, *J. Anthropological Research* 64:19–40 · Lewis 1972, *We, the Navigators* · Allen and Plourde 1999, USCG CG-D-08-99 · Soci et al. 2024, *QJRMS* 150:4014–4048 · [ERA5](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels) · [GLORYS12V1](https://data.marine.copernicus.eu/product/GLOBAL_MULTIYEAR_PHY_001_030/description) · [NOAA RONI](https://www.cpc.ncep.noaa.gov/data/indices/RONI.ascii.txt) · [PMIP4 past1000](https://gmd.copernicus.org/articles/10/4005/2017/gmd-10-4005-2017.html) · [GSHHG](https://github.com/GenericMappingTools/gshhg-gmt/releases) · [Hōkūleʻa 1976 timeline](https://mauinow.com/2026/04/24/50th-anniversary-of-the-historic-launch-of-hokule%CA%BBas-1976-maiden-voyage/) · [Hōkūleʻa specifications](https://en.wikipedia.org/wiki/H%C5%8Dk%C5%ABle%CA%BBa)
