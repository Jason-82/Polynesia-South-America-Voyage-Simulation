import itertools
import os
import time
from pathlib import Path

import numpy as np
import pytest
import yaml

from voyage_sim import catalog, geo
from voyage_sim.landfall import circle_ring

ROOT = Path(__file__).resolve().parents[1]
REAL_GSHHG = ROOT / "data" / "gshhg"
KM_PER_DEG = np.radians(geo.EARTH_RADIUS_KM)

# A synthetic world: a continent whose Pacific coast runs down 80W (its Atlantic
# side, at 60W, lies outside the Pacific mask), an atoll of twelve motu, a high
# island with a rock and a phantom copy (dropped by id), and islands off the
# coast at various distances.
CONFIG = """
domain: [-100.0, -60.0, -30.0, 10.0]
merge_km: 3.0
match_km: 20.0
continent_min_area_km2: 1.0e+5
drop_ids: [22]
pacific_mask: [[-100.5, -31.0], [-70.0, -31.0], [-70.0, 11.0], [-100.5, 11.0]]
groups:
  - name: Atolls
    region: polynesia
    kind: atoll
    height_m: 18
    source: canopy
    boxes: [[-95.0, -90.0, -12.0, -8.0]]
    islands:
      - {name: Big Atoll, ref: [-92.5, -10.0], height_m: 20, radius_km: 6.0, source: dem}
  - name: Highs
    region: offshore
    height_m: 50
    merge_km: 1.0
    boxes: [[-95.0, -90.0, -20.0, -15.0]]
    islands:
      - {name: Peak Island, ref: [PEAK_REF_LON, -17.0], height_m: 900, radius_km: 3.0, shift_to_ref: true}
      - {name: Lost Island, ref: [-93.5, -19.5], height_m: 300, radius_km: 1.5}
coast:
  relief_csv: relief.csv
  relief_source: relief table
  islet_drop_km: 15.0
  island_join_km: 150.0
  island_height_m: 44
  segments:
    - {name: "Coast: B", region: americas_other, north: -1.43, south: -2.0, height_m: 222, source: guess B}
    - {name: "Coast: A", region: americas, north: 1.0, south: -1.43, height_m: 111, source: guess A}
  islands:
    - {name: Coastal Isle, ref: [COASTAL_LON, -0.55], height_m: 77, radius_km: 2.0}
  placeholder:
    - [[-80.0, 2.0], [-80.5, -2.5]]
"""


def offset(lon, lat, east_km, north_km=0.0):
    """Point displaced east/north by km on the local tangent plane."""
    return float(lon + east_km / (KM_PER_DEG * np.cos(np.radians(lat)))), float(lat + north_km / KM_PER_DEG)


PEAK_CENTRE = (-92.0, -17.0)
PEAK_REF = offset(*PEAK_CENTRE, 2.5)          # GSHHG puts Peak Island 2.5 km west of its reference
COASTAL = offset(-80.0, -0.55, -40.0)         # 40 km off the coast
# Centres and radii (km) of the polygons that leave no landmass.
ISLET = (offset(-80.0, -0.2, -5.0), 0.5)      # wholly within 15 km of the coast
FAR = (offset(-80.0, -5.0, -300.0), 1.0)      # beyond island_join_km, outside every box
ATLANTIC = ((-65.0, -5.0), 2.0)               # outside the Pacific mask
OUTSIDE = ((-100.2, -5.0), 30.0)              # centroid outside the domain
PHANTOM = (offset(*PEAK_CENTRE, -12.0), 3.0)  # Peak Island's copy, id 22 in drop_ids


def relief_band(lat):
    """Synthetic relief of the 0.1 deg band centred on lat."""
    return round(300 + 100 * lat, 1)


# Bands without relief: 0.45 (Coast: A falls back to 111 m) and the two southernmost (Coast: B, 222 m).
RELIEF_LATS = [round(x, 2) for x in np.arange(0.95, -1.96, -0.1) if round(x, 2) not in (0.45, -1.85, -1.95)]


def shapes():
    """(ring, area_km2) of every synthetic GSHHG polygon."""
    lat = np.linspace(5.0, -25.0, 601)              # vertices every 0.05 deg, some exactly on band edges
    pacific = np.column_stack([np.full_like(lat, -80.0), lat])
    atlantic = np.column_stack([np.full_like(lat, -60.0), lat[::-1]])
    continent = np.vstack([pacific, atlantic, pacific[:1]])
    out = [(continent, 7.4e6)]
    motu = [circle_ring(*offset(-92.5, -10.0, 6 * np.sin(b), 6 * np.cos(b)), 0.4, n=16)
            for b in np.radians(np.arange(0, 360, 30))]   # 3.1 km apart: shorelines 2.3 km apart
    out += [(m, 0.5) for m in motu]
    out.append((circle_ring(*offset(-92.5, -10.0, 15.0), 0.4, n=16), 0.5))    # 9 km beyond the rim
    out.append((circle_ring(*PEAK_CENTRE, 3.0), 28.0))
    out.append((circle_ring(*offset(*PEAK_CENTRE, 0.0, -3.5), 0.2, n=12), 0.1))   # rock 0.3 km off its shore
    out.append((circle_ring(*ISLET[0], ISLET[1], n=16), 0.8))
    out.append((circle_ring(*COASTAL, 2.0), 12.0))
    out.append((circle_ring(*offset(-80.0, -1.8, -60.0), 1.0), 3.0))          # unnamed coastal island
    out.append((circle_ring(*FAR[0], FAR[1]), 3.0))
    out.append((circle_ring(*ATLANTIC[0], ATLANTIC[1]), 12.0))
    out.append((circle_ring(*OUTSIDE[0], OUTSIDE[1]), 2800.0))
    out.append((circle_ring(*PHANTOM[0], PHANTOM[1]), 28.0))
    assert len(out) == 23                                                      # the phantom is id 22
    return out


def write_world(root):
    """A GSHHG-like release under ``root``; returns the shapefile path."""
    shapefile = pytest.importorskip("shapefile")
    shp = root / catalog.GSHHG_SHAPEFILE
    shp.parent.mkdir(parents=True, exist_ok=True)
    with shapefile.Writer(str(shp.with_suffix("")), shapeType=shapefile.POLYGON) as writer:
        writer.field("id", "C", 80)
        writer.field("level", "N", 9)
        writer.field("area", "N", 24, 15)
        for k, (ring, area) in enumerate(shapes()):
            writer.poly([np.asarray(ring).tolist()])
            writer.record(str(k), 1, area)
    return shp


def write_config(root, **edits):
    text = CONFIG.replace("PEAK_REF_LON", repr(PEAK_REF[0])).replace("COASTAL_LON", repr(COASTAL[0]))
    for old, new in edits.items():
        assert old in text
        text = text.replace(old, new)
    (root / "relief.csv").write_text("# synthetic\nlat,max5\n" + "".join(f"{x},{relief_band(x)}\n"
                                                                       for x in RELIEF_LATS))
    path = root / "landmasses.yaml"
    path.write_text(text)
    return path


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("world")
    write_world(root)
    return root, write_config(root)


@pytest.fixture(scope="module")
def built(world):
    root, config = world
    return catalog.build_catalog(config, gshhg_dir=root)


def by_name(cat):
    return {lm.name: lm for lm in cat.landmasses}


# --- GSHHG mode ---------------------------------------------------------------------

def test_polygons_are_sorted_by_fate(built):
    assert built.counts == {
        "polygons": 23, "dropped_by_id": 1, "continental": 1, "outside_domain": 1, "outside_mask": 1,
        "coastal_islets_dropped": 1, "unassigned_dropped": 1, "in_groups": 15, "coastal_islands": 2,
        "merged": 12,                 # eleven motu into the first, the rock into Peak Island
        "named_not_found": 1,         # Lost Island
        "landmasses_island": 4, "landmasses_atoll": 2, "landmasses_coast": 31}
    assert built.group_region == {"Atolls": "polynesia", "Highs": "offshore", "Coast: A": "americas",
                                  "Coast: B": "americas_other"}
    assert built.groups_in("americas", "polynesia") == ["Atolls", "Coast: A"]
    assert built.groups_in("other") == []
    with pytest.raises(ValueError, match="unknown regions"):
        built.groups_in("pacific")
    names = [lm.name for lm in built.landmasses]
    assert len(set(names)) == len(names)
    assert names[:4] == ["Big Atoll", "Atolls #1", "Peak Island", "Lost Island"]


def test_motu_merge_into_one_atoll(built):
    lms = by_name(built)
    atoll, islet = lms["Big Atoll"], lms["Atolls #1"]
    assert (atoll.group, atoll.kind, atoll.height_m, len(atoll.rings)) == ("Atolls", "atoll", 20.0, 12)
    # The islet 9 km beyond the rim stays apart and takes the group's defaults.
    assert (islet.group, islet.kind, islet.height_m, islet.source, len(islet.rings)) == (
        "Atolls", "atoll", 18.0, "canopy", 1)
    row = built.named.set_index("name").loc["Big Atoll"]
    # The reference is the lagoon centre: no motu contains it; the nearest rim is 6 - 0.4 km away.
    assert row.found and not row.inside and row.landmass == "Big Atoll"
    assert row.nearest_km == pytest.approx(5.6, abs=0.3)
    assert row.centroid_km < 0.05


def test_named_island_is_shifted_onto_its_reference(built):
    lms = by_name(built)
    peak = lms["Peak Island"]
    assert (peak.group, peak.kind, peak.height_m, len(peak.rings)) == ("Highs", "island", 900.0, 2)
    row = built.named.set_index("name").loc["Peak Island"]
    assert row.found and row.inside and row.shifted and row.nearest_km == 0.0
    assert row.centroid_km == pytest.approx(2.5, abs=0.05)       # the offset found, before the shift
    _, lon, lat = catalog._area_centroid(peak.rings[:1])
    assert geo.haversine_km(lon, lat, *PEAK_REF) < 0.05
    # The rock moves with its island.
    moved = [ring - original for ring, original in zip(peak.rings, (circle_ring(*PEAK_CENTRE, 3.0),
             circle_ring(*offset(*PEAK_CENTRE, 0.0, -3.5), 0.2, n=12)))]
    for d in moved:
        np.testing.assert_allclose(d, moved[0][:1].repeat(len(d), axis=0), atol=1e-9)


def test_unmatched_named_island_keeps_its_circle(built):
    lost = by_name(built)["Lost Island"]
    assert lost.group == "Highs" and lost.source.startswith("placeholder circle")
    np.testing.assert_allclose(geo.haversine_km(-93.5, -19.5, *lost.rings[0].T), 1.5, rtol=1e-9)
    row = built.named.set_index("name").loc["Lost Island"]
    assert not row.found and np.isnan(row.centroid_km) and row.nearest_km > 20


def test_coast_is_cut_into_pieces_with_relief_heights(built):
    pieces = [lm for lm in built.landmasses if lm.kind == "coast"]
    a = [lm for lm in pieces if lm.group == "Coast: A"]
    b = [lm for lm in pieces if lm.group == "Coast: B"]
    assert len(a) == 25 and len(b) == 6
    assert a[0].name == "Coast: A 1.00N-0.90N" and a[-1].name == "Coast: A 1.40S-1.43S"
    assert b[0].name == "Coast: B 1.43S-1.50S" and b[-1].name == "Coast: B 1.90S-2.00S"
    for lm in pieces:
        verts = np.vstack(lm.rings)
        # Only the Pacific side, inside the mask; the Atlantic coast at 60W is not coast.
        np.testing.assert_array_equal(verts[:, 0], -80.0)
        hi, lo = (float(x[:-1]) * (1 if x[-1] == "N" else -1) for x in lm.name.split()[-1].split("-"))
        assert verts[:, 1].max() == pytest.approx(hi) and verts[:, 1].min() == pytest.approx(lo)
    lms = by_name(built)
    assert lms["Coast: A 1.00N-0.90N"].height_m == relief_band(0.95)
    assert lms["Coast: A 1.00N-0.90N"].source == "relief table"
    assert (lms["Coast: A 0.50N-0.40N"].height_m, lms["Coast: A 0.50N-0.40N"].source) == (111.0, "guess A")
    # The band 1.4-1.5S is split at the segment boundary; both parts take its relief.
    assert lms["Coast: A 1.40S-1.43S"].height_m == lms["Coast: B 1.43S-1.50S"].height_m == relief_band(-1.45)
    assert lms["Coast: B 1.90S-2.00S"].height_m == 222.0
    # Pieces meet: the coast between 1N and 2S is covered once, without gaps.
    spans = sorted((np.vstack(lm.rings)[:, 1].min(), np.vstack(lm.rings)[:, 1].max()) for lm in pieces)
    assert spans[0][0] == pytest.approx(-2.0) and spans[-1][1] == pytest.approx(1.0)
    for (_, top), (bottom, _) in itertools.pairwise(spans):
        assert bottom == pytest.approx(top)


def test_coastal_islands_join_or_are_dropped(built):
    lms = by_name(built)
    isle = lms["Coastal Isle"]
    assert (isle.group, isle.kind, isle.height_m) == ("Coast: A", "island", 77.0)
    unnamed = lms["Coast: B #1"]
    assert (unnamed.group, unnamed.kind, unnamed.height_m) == ("Coast: B", "island", 44.0)
    # Segment order: pieces north to south, then the segment's islands.
    group_b = [lm.name for lm in built.landmasses if lm.group == "Coast: B"]
    assert group_b[-1] == "Coast: B #1"
    # Nothing is left of the islet 5 km off the coast, the far island, the Atlantic one, the one outside
    # the domain or the phantom dropped by id.
    verts = np.vstack([r for lm in built.landmasses if lm.kind != "coast" for r in lm.rings])
    for (lon, lat), radius in (ISLET, FAR, ATLANTIC, OUTSIDE, PHANTOM):
        assert geo.haversine_km(lon, lat, verts[:, 0], verts[:, 1]).min() > radius + 5.0


def test_unknown_drop_ids_are_reported(tmp_path, caplog):
    write_world(tmp_path)
    config = write_config(tmp_path, **{"drop_ids: [22]": "drop_ids: [22, '999']"})
    cat = catalog.build_catalog(config, gshhg_dir=tmp_path)
    assert cat.counts["dropped_by_id"] == 1
    assert "drop_ids ['999']: no such polygon" in caplog.text


def test_cut_shares_crossing_points():
    cuts = np.array([0.0, 0.1, 0.2, 0.3])
    # Vertices on a cut, an edge across two cuts, and a return across the same cut.
    line = np.array([[0.0, 0.05], [0.1, 0.1], [0.2, 0.35], [0.3, 0.25], [0.4, 0.28]])
    bins, pieces = catalog._cut(line, cuts)
    assert list(bins) == [1, 2, 3, 4, 3]
    np.testing.assert_allclose(pieces[0], [[0.0, 0.05], [0.1, 0.1]])
    np.testing.assert_allclose(pieces[2], [[0.14, 0.2], [0.18, 0.3]])
    for a, b in itertools.pairwise(pieces):
        np.testing.assert_array_equal(a[-1], b[0])


# --- placeholder mode and the cache -----------------------------------------------------

def test_placeholder_without_gshhg(world, tmp_path, caplog):
    _, config = world
    cat = catalog.build_catalog(config)
    missing = catalog.build_catalog(config, gshhg_dir=tmp_path / "nowhere")
    assert "placeholder" in caplog.text
    assert [lm.name for lm in cat.landmasses] == [lm.name for lm in missing.landmasses]
    lms = by_name(cat)
    for name, (lon, lat), radius in (("Big Atoll", (-92.5, -10.0), 6.0), ("Peak Island", PEAK_REF, 3.0),
                                     ("Coastal Isle", COASTAL, 2.0)):
        np.testing.assert_allclose(geo.haversine_km(lon, lat, *lms[name].rings[0].T), radius, rtol=1e-9)
    assert lms["Coastal Isle"].group == "Coast: A"
    pieces = [lm for lm in cat.landmasses if lm.kind == "coast"]
    assert len(pieces) == 31 and pieces[0].name == "Coast: A 1.00N-0.90N"
    assert lms["Coast: A 1.00N-0.90N"].height_m == relief_band(0.95)
    assert cat.group_region == {"Atolls": "polynesia", "Highs": "offshore", "Coast: A": "americas",
                                "Coast: B": "americas_other"}
    assert cat.named.empty and cat.counts == {"landmasses_island": 3, "landmasses_atoll": 1,
                                              "landmasses_coast": 31}


def test_cache_is_reused_until_an_input_changes(world, tmp_path, monkeypatch):
    root, config = world
    first = catalog.build_catalog(config, gshhg_dir=root, cache_dir=tmp_path)
    assert len(list(tmp_path.glob("catalog-*.pkl"))) == 1

    def fail(*_):
        raise AssertionError("rebuilt")
    monkeypatch.setattr(catalog, "_from_gshhg", fail)
    again = catalog.build_catalog(config, gshhg_dir=root, cache_dir=tmp_path)
    assert [lm.name for lm in again.landmasses] == [lm.name for lm in first.landmasses]
    np.testing.assert_array_equal(again.landmasses[0].rings[0], first.landmasses[0].rings[0])
    assert again.named.equals(first.named)

    shp = root / catalog.GSHHG_SHAPEFILE
    stat = shp.stat()
    os.utime(shp, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10**9))
    try:
        with pytest.raises(AssertionError, match="rebuilt"):
            catalog.build_catalog(config, gshhg_dir=root, cache_dir=tmp_path)
    finally:
        os.utime(shp, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    edited = write_config(tmp_path, **{"island_height_m: 44": "island_height_m: 45"})
    with pytest.raises(AssertionError, match="rebuilt"):
        catalog.build_catalog(edited, gshhg_dir=root, cache_dir=tmp_path)


@pytest.mark.parametrize("edits, match", [
    ({"region: offshore": "region: pacific"}, "region"),
    ({"merge_km: 1.0\n": "merge_km: 1.0\n    colour: red\n"}, "unknown keys"),
    ({"radius_km: 1.5}": "radius_km: 1.5, ref_type: peak, shift_to_ref: true}"}, "shift_to_ref"),
    ({"name: Lost Island": "name: Big Atoll"}, "duplicate island"),
    ({"north: -1.43, south: -2.0": "north: -1.0, south: -2.0"}, "overlap"),
    ({"height_m: 222, source: guess B": "source: guess B"}, "no relief data"),
    ({"boxes: [[-95.0, -90.0, -12.0, -8.0]]": "boxes: [[-90.0, -95.0, -12.0, -8.0]]"}, "boxes"),
    ({"drop_ids: [22]": "drop_ids: 22"}, "drop_ids"),
])
def test_config_errors(tmp_path, edits, match):
    with pytest.raises(ValueError, match=match):
        catalog.build_catalog(write_config(tmp_path, **edits))


# --- the project's configs ------------------------------------------------------------

def test_project_configs_placeholder():
    cat = catalog.build_catalog(ROOT / "configs" / "landmasses.yaml")
    assert cat.groups_in("americas") == ["Coast: Colombia", "Coast: Ecuador", "Coast: Northern Peru",
                                         "Coast: Southern Peru", "Coast: Northern Chile", "Coast: Central Chile"]
    assert set(cat.groups_in("polynesia")) == {"South Marquesas", "North Marquesas", "Tuamotus", "Gambier",
                                               "Pitcairn Islands", "Rapa Nui", "Society Islands", "Austral Islands"}
    # Named islands default to their group's kind: a high island in an atoll group must say kind: island.
    for lm in cat.landmasses:
        assert lm.kind != "atoll" or lm.height_m <= 40, lm.name
    with open(ROOT / "configs" / "places.yaml", encoding="utf-8") as fh:
        places = yaml.safe_load(fh)
    targets = {t["name"] for t in places["targets"]}
    for section in ("departures", "targets", "validation"):
        for place in places[section]:
            assert place["group"] in cat.group_region, place
            if section == "validation":
                assert place["target"] in targets
                assert np.isfinite(geo.to_epoch_hours(place["start"]))


@pytest.mark.skipif(not (REAL_GSHHG / catalog.GSHHG_SHAPEFILE).is_file(), reason="GSHHG not downloaded")
def test_real_gshhg_catalog():
    import pandas as pd

    from voyage_sim.landfall import SightingRaster, SightingRule

    config = ROOT / "configs" / "landmasses.yaml"
    start = time.perf_counter()
    cat = catalog.build_catalog(config, gshhg_dir=REAL_GSHHG)
    build_s = time.perf_counter() - start
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(f"\ncatalog built in {build_s:.1f} s: {cat.counts}")
        print(cat.named[["name", "group", "landmass", "inside", "nearest_km", "centroid_km", "shifted"]]
              .round(2).to_string())
    assert build_s < 60

    doc = yaml.safe_load(config.read_text(encoding="utf-8"))
    expected = {isl["name"]: g["name"] for g in doc["groups"] for isl in g.get("islands", [])}
    expected.update({"Coiba": "Coast: Central America", "Gorgona": "Coast: Colombia",
                     "Isla de la Plata": "Coast: Ecuador", "Lobos de Tierra": "Coast: Northern Peru",
                     "Lobos de Afuera": "Coast: Northern Peru", "Isla Santa María": "Coast: Central Chile",
                     "Isla Mocha": "Coast: Central Chile", "Chiloé": "Coast: Southern Chile"})
    named = cat.named.set_index("name")
    assert named["group"].to_dict() == expected
    assert set(named.index[~named["found"].astype(bool)]) == {"Temoe", "Roca Partida"}   # too small for GSHHG
    assert named.loc["Sala y Gómez", "centroid_km"] == pytest.approx(10.4, abs=1.0)
    assert named.loc["Sala y Gómez", "shifted"]
    # GSHHG's phantom copies (Sala y Gómez 19 km WNW, San Félix 4 km N, Santa Clara 5 km W) are dropped.
    assert cat.counts["dropped_by_id"] == 3
    assert [lm.name for lm in cat.landmasses if lm.group == "Sala y Gómez"] == ["Sala y Gómez"]
    lms = by_name(cat)
    for name, (lon, lat) in (("San Félix", (-80.085, -26.258)), ("Robinson Crusoe", (-78.996, -33.696))):
        verts = np.vstack(lms[name].rings)
        assert geo.haversine_km(lon, lat, verts[:, 0], verts[:, 1]).min() > 2.0   # 0.9 and 0.2 km if kept
    for name, group in expected.items():
        assert lms[named.loc[name, "landmass"]].group == group

    pieces = np.vstack([r for lm in cat.landmasses if lm.kind == "coast" for r in lm.rings])
    assert pieces[:, 1].min() == pytest.approx(-50.0, abs=0.05) and pieces[:, 1].max() == pytest.approx(30.0, abs=0.05)
    # No Caribbean or Gulf of Mexico shore.
    assert pieces[pieces[:, 1] > 7.2, 0].max() < -77.5
    assert pieces[pieces[:, 1] > 17.0, 0].max() < -100.0
    assert len({lm.name for lm in cat.landmasses}) == len(cat.landmasses)

    start = time.perf_counter()
    raster = SightingRaster.build(cat.landmasses, SightingRule("main", bird_km=37.0), (-165.0, -65.0, -50.0, 30.0),
                                  res_deg=0.05, cap_km=100.0)
    raster_s = time.perf_counter() - start
    print(f"sighting raster for {len(cat.landmasses)} landmasses, full domain at 0.05 deg: {raster_s:.1f} s")
    assert raster_s < 180
    margin, index = raster.query(np.array([-109.355, -138.652]), np.array([-27.119, -10.485]))
    assert np.all(margin < 0)
    assert [cat.landmasses[i].name for i in index] == ["Rapa Nui", "Fatu Hiva"]
