"""The land in the study domain: landmasses with groups, regions and sighting heights.

The landfall detector needs every shoreline a voyage could sight, each with a
group (home exclusion and destinations work by group), a region (which groups
are the far side of the ocean) and a sighting height. ``build_catalog`` makes
them from ``configs/landmasses.yaml`` and, when downloaded, the full-resolution
GSHHG level-1 shorelines (the land/ocean boundary):

- Continental shoreline inside the config's Pacific mask (west of the
  continental divide of the Americas, so no Caribbean or Atlantic shore) is the
  Pacific coast. It is cut into coast segments by latitude and each segment
  into 0.1 deg latitude pieces, whose height is the highest ground within 5 km
  of the shore (``configs/coast_relief.csv``, Copernicus GLO-90); bands without
  relief data take the segment's height.
- Polygons listed in the config's ``drop_ids`` are left out: phantom copies
  of islands that GSHHG holds twice, from two source databases.
- Every other polygon is an island. One inside a group's box belongs to that
  group. Otherwise, inside the mask and near the Pacific coast, it is a coastal
  island and joins the coast segment at its latitude, unless it lies wholly
  within a few km of the continent, inside the coast's sighting zone anyway,
  and is dropped. Anything else is dropped and counted.
- Polygons of one pool (a group, or the coastal islands) whose shorelines come
  within ``merge_km`` of each other become one landmass: the motu of an atoll,
  an island and its rocks.
- Named islands carry literature heights and a reference position. Each is
  matched to the landmass containing (else nearest) its reference; the offset
  is reported in ``Catalog.named`` and, with ``shift_to_ref``, the shoreline is
  moved so its centroid sits on the reference (SPEC: shoreline datasets
  misplace some small islands by kilometres). Unnamed landmasses take their
  group's height.

Without GSHHG the catalog is a placeholder: named islands as circles of their
``radius_km`` and the coast from a coarse polyline in the config, cut into the
same pieces. Good enough for synthetic runs and tests, not for results.
"""

from __future__ import annotations

import hashlib
import itertools
import logging
import os
import pickle
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from voyage_sim import geo, landfall
from voyage_sim.landfall import LANDMASS_KINDS, Landmass, circle_ring

log = logging.getLogger(__name__)

REGIONS = ("polynesia", "americas", "americas_other", "offshore", "other")
GSHHG_SHAPEFILE = Path("GSHHS_shp", "f", "GSHHS_f_L1.shp")   # inside gshhg_dir
SAMPLE_KM = 0.5   # shoreline sampling step for merging and matching
_KM_PER_DEG = np.radians(geo.EARTH_RADIUS_KM)
_NAMED_COLUMNS = ["name", "group", "landmass", "found", "inside", "nearest_km", "centroid_km", "shifted",
                  "ref_type", "ref_lon", "ref_lat", "centroid_lon", "centroid_lat"]


@dataclass
class Catalog:
    """Landmasses and the region of each group; groups in the order the landfall detector numbers them.

    ``named`` (GSHHG only) has one row per named island: the landmass it
    matched, whether its reference lies inside that shoreline, the distance
    from the reference to the shoreline and to its centroid before any shift,
    and whether the shoreline was shifted onto the reference. ``counts``
    tallies the GSHHG polygons by fate and the landmasses by kind.
    """

    landmasses: list[Landmass]
    group_region: dict[str, str]   # group -> one of REGIONS
    named: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=_NAMED_COLUMNS))
    counts: dict[str, int] = field(default_factory=dict)

    def groups_in(self, *regions) -> list[str]:
        """Groups whose region is any of ``regions``, in landmass order."""
        if unknown := set(regions) - set(REGIONS):
            raise ValueError(f"unknown regions {sorted(unknown)}; known: {REGIONS}")
        return [g for g, r in self.group_region.items() if r in regions]


def build_catalog(config_path, gshhg_dir=None, cache_dir=None) -> Catalog:
    """Build the catalog from ``config_path`` and GSHHG under ``gshhg_dir`` (placeholder without it).

    ``gshhg_dir`` is the unpacked GSHHG shapefile release (containing
    ``GSHHS_shp/f/GSHHS_f_L1.shp``). With ``cache_dir``, the catalog is pickled
    there, keyed by the config, the relief table, the source of this module and
    of ``landfall`` (whose ``Landmass`` objects it pickles) and the shapefile's
    size and mtime, and reused while those are unchanged.
    """
    config_path = Path(config_path)
    cfg = _load_config(config_path)
    shp = None
    if gshhg_dir is not None:
        shp = Path(gshhg_dir) / GSHHG_SHAPEFILE
        if not shp.is_file():
            log.warning("no GSHHG shapefile at %s; building the placeholder catalog", shp)
            shp = None
    cache = None
    if cache_dir is not None:
        cache = Path(cache_dir) / f"catalog-{_cache_key(config_path, cfg, shp)}.pkl"
        if cache.is_file():
            with open(cache, "rb") as fh:
                return pickle.load(fh)
    catalog = _from_gshhg(cfg, shp) if shp else _placeholder(cfg)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache.with_suffix(f".{os.getpid()}.tmp")
        with open(tmp, "wb") as fh:
            pickle.dump(catalog, fh, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, cache)
    return catalog


def _cache_key(config_path, cfg, shp):
    h = hashlib.sha256(Path(__file__).read_bytes())
    h.update(Path(landfall.__file__).read_bytes())
    h.update(config_path.read_bytes())
    if cfg["relief_path"] is not None:
        h.update(cfg["relief_path"].read_bytes())
    if shp is None:
        h.update(b"placeholder")
    else:
        st = shp.stat()
        h.update(f"{shp.resolve()}:{st.st_size}:{st.st_mtime_ns}".encode())
    return h.hexdigest()[:20]


# --- config -------------------------------------------------------------------

def _fields(cfg, what, required=(), **defaults):
    """Check a config mapping's keys; return a dict with defaults filled in."""
    if not isinstance(cfg, Mapping):
        raise ValueError(f"{what} must be a mapping")
    if unknown := set(cfg) - set(required) - set(defaults):
        raise ValueError(f"{what}: unknown keys {sorted(unknown)}; allowed: {sorted({*required, *defaults})}")
    if missing := set(required) - set(cfg):
        raise ValueError(f"{what}: missing keys {sorted(missing)}")
    return {**defaults, **cfg}


def _lonlat(value, what):
    pt = np.asarray(value, dtype=float)
    if pt.shape != (2,) or not np.all(np.isfinite(pt)) or abs(pt[1]) > 90:
        raise ValueError(f"{what} must be [lon, lat]")
    return float(geo.wrap_lon(pt[0])), float(pt[1])


def _path(value, what, closed=False):
    pts = np.asarray(value, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < (3 if closed else 2) or not np.all(np.isfinite(pts)):
        raise ValueError(f"{what} must be a list of [lon, lat] pairs")
    return _closed(pts) if closed else pts


def _island(cfg, what):
    isl = _fields(cfg, what, ("name", "ref", "height_m", "radius_km"), source="", kind=None,
                  ref_type="centroid", shift_to_ref=False)
    what = f"{what} {isl['name']!r}"
    isl["ref"] = _lonlat(isl["ref"], f"{what} ref")
    isl["height_m"], isl["radius_km"] = float(isl["height_m"]), float(isl["radius_km"])
    if not (isl["height_m"] >= 0 and isl["radius_km"] > 0):
        raise ValueError(f"{what}: need height_m >= 0 and radius_km > 0")
    if isl["kind"] not in (None, "island", "atoll"):
        raise ValueError(f"{what}: kind must be island or atoll")
    if isl["ref_type"] not in ("centroid", "peak"):
        raise ValueError(f"{what}: ref_type must be centroid or peak")
    if isl["shift_to_ref"] and isl["ref_type"] != "centroid":
        raise ValueError(f"{what}: shift_to_ref needs a centroid reference")
    return isl


def _load_config(path):
    """Read and check ``landmasses.yaml`` (schema at the top of that file)."""
    with open(path, encoding="utf-8") as fh:
        doc = yaml.safe_load(fh)
    cfg = _fields(doc, str(path), ("domain", "pacific_mask", "groups", "coast"),
                  merge_km=3.0, match_km=20.0, continent_min_area_km2=1.0e6, drop_ids=())
    for key in ("merge_km", "match_km", "continent_min_area_km2"):
        cfg[key] = float(cfg[key])
    if not isinstance(cfg["drop_ids"], (list, tuple)):
        raise ValueError("drop_ids must be a list of GSHHG polygon ids")
    cfg["drop_ids"] = {str(i) for i in cfg["drop_ids"]}
    dom = np.asarray(cfg["domain"], dtype=float)
    if dom.shape != (4,) or not (dom[0] < dom[1] and dom[2] < dom[3]):
        raise ValueError("domain must be [lon_min, lon_max, lat_min, lat_max]")
    cfg["domain"] = tuple(dom)
    cfg["pacific_mask"] = _path(cfg["pacific_mask"], "pacific_mask", closed=True)
    if not (cfg["merge_km"] >= 0 and cfg["match_km"] > 0):
        raise ValueError("need merge_km >= 0 and match_km > 0")

    groups = []
    for k, g in enumerate(cfg["groups"]):
        g = _fields(g, f"group #{k}", ("name", "region", "boxes", "height_m"), kind="island", source="",
                    merge_km=None, islands=())
        what = f"group {g['name']!r}"
        if g["region"] not in REGIONS or g["kind"] not in ("island", "atoll"):
            raise ValueError(f"{what}: region must be one of {REGIONS} and kind island or atoll")
        boxes = np.asarray(g["boxes"], dtype=float)
        if boxes.ndim != 2 or boxes.shape[1] != 4 or not np.all((boxes[:, 0] < boxes[:, 1])
                                                                 & (boxes[:, 2] < boxes[:, 3])):
            raise ValueError(f"{what}: boxes must be [[lon_min, lon_max, lat_min, lat_max], ...]")
        g["boxes"], g["height_m"] = boxes, float(g["height_m"])
        g["merge_km"] = cfg["merge_km"] if g["merge_km"] is None else float(g["merge_km"])
        g["islands"] = [_island(isl, f"{what} island") for isl in g["islands"]]
        groups.append(g)
    cfg["groups"] = groups

    coast = _fields(cfg["coast"], "coast", ("segments", "placeholder"), piece_deg=0.1, relief_csv=None,
                    relief_column="max5", relief_source="", islet_drop_km=15.0, island_join_km=150.0,
                    island_height_m=50.0, island_source="", islands=())
    for key in ("piece_deg", "islet_drop_km", "island_join_km", "island_height_m"):
        coast[key] = float(coast[key])
    if not (coast["piece_deg"] > 0 and 0 <= coast["islet_drop_km"] <= coast["island_join_km"]):
        raise ValueError("coast: need piece_deg > 0 and 0 <= islet_drop_km <= island_join_km")
    segments = []
    for k, s in enumerate(coast["segments"]):
        s = _fields(s, f"coast segment #{k}", ("name", "region", "north", "south"), height_m=None, source="")
        s["north"], s["south"] = float(s["north"]), float(s["south"])
        s["height_m"] = None if s["height_m"] is None else float(s["height_m"])
        if s["region"] not in REGIONS or not s["south"] < s["north"]:
            raise ValueError(f"coast segment {s['name']!r}: region must be one of {REGIONS} and south < north")
        segments.append(s)
    segments.sort(key=lambda s: -s["north"])
    if any(a["south"] < b["north"] for a, b in itertools.pairwise(segments)):
        raise ValueError("coast segments overlap")
    coast["segments"] = segments
    coast["islands"] = [_island(isl, "coast island") for isl in coast["islands"]]
    coast["placeholder"] = [_path(line, "coast placeholder line") for line in coast["placeholder"]]
    for what, names in (("group", [g["name"] for g in groups] + [s["name"] for s in segments]),
                        ("island", [i["name"] for g in groups for i in g["islands"]]
                         + [i["name"] for i in coast["islands"]])):
        if dupes := sorted({n for n in names if names.count(n) > 1}):
            raise ValueError(f"duplicate {what} names {dupes}")

    cfg["relief_path"] = None
    cfg["relief"] = {}
    if coast["relief_csv"] is not None:
        cfg["relief_path"] = Path(path).parent / coast["relief_csv"]
        table = pd.read_csv(cfg["relief_path"], comment="#")
        keep = table[coast["relief_column"]].notna().to_numpy()
        bands = _band(table["lat"].to_numpy()[keep], coast["piece_deg"])
        cfg["relief"] = dict(zip(bands.tolist(), table[coast["relief_column"]].to_numpy(float)[keep]))
    cfg["coast"] = coast
    return cfg


def _band(lat, piece_deg):
    """Index of the latitude band of width ``piece_deg`` containing ``lat`` (band k: [k, k+1) * piece_deg)."""
    return np.floor(np.round(np.asarray(lat, dtype=float) / piece_deg, 6)).astype(np.int64)


# --- geometry -----------------------------------------------------------------

def _closed(ring):
    ring = np.asarray(ring, dtype=float)
    return ring if np.array_equal(ring[0], ring[-1]) else np.vstack([ring, ring[:1]])


def _inside(poly, lon, lat):
    """Even-odd test of points against a closed polygon; vectorized over the points."""
    lon, lat = np.asarray(lon, dtype=float), np.asarray(lat, dtype=float)
    inside = np.zeros(lon.shape, dtype=bool)
    with np.errstate(divide="ignore", invalid="ignore"):
        for (xa, ya), (xb, yb) in itertools.pairwise(poly):
            spans = (ya > lat) != (yb > lat)
            inside ^= spans & (lon < xa + (lat - ya) * (xb - xa) / (yb - ya))
    return inside


def _in_box(lon, lat, box):
    lon_min, lon_max, lat_min, lat_max = box
    return (lon >= lon_min) & (lon <= lon_max) & (lat >= lat_min) & (lat <= lat_max)


def _area_centroid(rings):
    """Total area (km2) and area-weighted centroid (lon, lat) of closed rings.

    Planar shoelace on a local equirectangular projection per ring: exact
    enough for islands; degenerate rings count by their vertex mean.
    """
    areas, cents = [], []
    for ring in rings:
        lon0, lat0 = ring[0]
        kx = np.cos(np.radians(lat0)) * _KM_PER_DEG
        x, y = geo.angle_diff(ring[:, 0], lon0) * kx, (ring[:, 1] - lat0) * _KM_PER_DEG
        cross = x[:-1] * y[1:] - x[1:] * y[:-1]
        a = 0.5 * cross.sum()
        if abs(a) < 1e-9:
            areas.append(0.0)
            cents.append(ring.mean(axis=0))
            continue
        cx, cy = ((x[:-1] + x[1:]) * cross).sum() / (6 * a), ((y[:-1] + y[1:]) * cross).sum() / (6 * a)
        areas.append(abs(a))
        cents.append((lon0 + cx / kx, lat0 + cy / _KM_PER_DEG))
    areas, cents = np.array(areas), np.array(cents)
    weights = areas if areas.sum() > 0 else np.ones(len(areas))
    lon, lat = np.average(cents, axis=0, weights=weights)
    return areas.sum(), float(lon), float(lat)


def _runs(ring, keep):
    """Contiguous runs of a closed ring's vertices where ``keep`` holds, as open polylines."""
    if keep.all():
        return [ring]
    # Start at a dropped vertex so that no run wraps past the end of the ring.
    start = int(np.argmin(keep[:-1]))
    ring, keep = np.roll(ring[:-1], -start, axis=0), np.roll(keep[:-1], -start)
    edges = np.diff(np.concatenate([[0], keep.astype(np.int8), [0]]))
    return [ring[a:b] for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))]


def _cut(line, cuts):
    """Split an open polyline where it crosses the latitudes ``cuts`` (ascending).

    Crossing points are interpolated and shared by the pieces on either side.
    Returns ``(bins, pieces)``: piece k lies in bin ``bins[k]``, the latitudes
    between ``cuts[bins[k] - 1]`` and ``cuts[bins[k]]`` (bin 0 is south of all
    cuts, bin ``len(cuts)`` north of them).
    """
    if len(line) == 1:
        return np.searchsorted(cuts, line[:, 1], side="right"), [line]
    lat = line[:, 1]
    b = np.searchsorted(cuts, lat, side="right")
    n = np.abs(np.diff(b))                     # cuts crossed by each edge
    edge = np.repeat(np.arange(len(n)), n)
    k = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
    up = b[edge + 1] > b[edge]
    m = np.where(up, b[edge] + k, b[edge] - 1 - k)   # which cut, in travel order
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.clip((cuts[m] - lat[edge]) / (lat[edge + 1] - lat[edge]), 0.0, 1.0)
    cross = line[edge] + t[:, None] * (line[edge + 1] - line[edge])
    cross[:, 1] = cuts[m]
    pos = np.concatenate([np.arange(len(line), dtype=float), edge + (k + 1) / (n[edge] + 1)])
    order = np.argsort(pos, kind="stable")
    pts = np.vstack([line, cross])[order]
    is_cross = np.concatenate([np.zeros(len(line), bool), np.ones(len(cross), bool)])[order]
    ends = np.flatnonzero(is_cross)
    pieces = [pts[a:e + 1] for a, e in zip(np.r_[0, ends], np.r_[ends, len(pts) - 1])]
    # A piece's bin is that of its first segment's midpoint: interior vertices may sit on a cut.
    mids = np.array([0.5 * (p[0, 1] + p[1, 1]) for p in pieces])
    return np.searchsorted(cuts, mids, side="right"), pieces


def _sample(ring, step_km):
    """Points evenly spaced along a ring, at most ``step_km`` apart."""
    seg = geo.haversine_km(ring[:-1, 0], ring[:-1, 1], ring[1:, 0], ring[1:, 1])
    s = np.concatenate([[0.0], np.cumsum(seg)])
    at = np.linspace(0.0, s[-1], max(int(np.ceil(s[-1] / step_km)), 1) + 1)
    return np.column_stack([np.interp(at, s, ring[:, 0]), np.interp(at, s, ring[:, 1])])


def _xyz(pts):
    pts = np.asarray(pts, dtype=float).reshape(-1, 2)
    return geo.lonlat_to_xyz(pts[:, 0], pts[:, 1])


# --- the Pacific coast ----------------------------------------------------------

def _lat_label(lat):
    return f"{abs(lat):.2f}{'S' if lat < 0 else 'N'}"


def _coast_pieces(lines, cfg):
    """Cut coast polylines into pieces by segment and latitude band -> {segment: [Landmass]} north to south."""
    coast, relief = cfg["coast"], cfg["relief"]
    segments, piece = coast["segments"], coast["piece_deg"]
    south, north = segments[-1]["south"], segments[0]["north"]
    grid = np.arange(np.ceil(np.round(south / piece, 6)), np.floor(np.round(north / piece, 6)) + 1) * piece
    bounds = [x for s in segments for x in (s["north"], s["south"])]
    cuts = np.unique(np.round(np.concatenate([grid, bounds]), 9))
    parts = defaultdict(list)
    for line in lines:
        for b, part in zip(*_cut(line, cuts)):
            if 0 < b < len(cuts):
                parts[int(b)].append(part)

    out = {s["name"]: [] for s in segments}
    for b in sorted(parts, reverse=True):
        lo, hi = cuts[b - 1], cuts[b]
        seg = next((s for s in segments if s["south"] <= lo and hi <= s["north"]), None)
        if seg is None:   # a gap between segments
            continue
        band = int(_band(0.5 * (lo + hi), piece))
        if band in relief:
            height, source = relief[band], coast["relief_source"]
        elif seg["height_m"] is not None:
            height, source = seg["height_m"], seg["source"]
        else:
            raise ValueError(f"{seg['name']}: no relief data for {_lat_label(hi)}-{_lat_label(lo)} "
                             "and no segment height_m")
        out[seg["name"]].append(Landmass(f"{seg['name']} {_lat_label(hi)}-{_lat_label(lo)}", seg["name"],
                                         height, parts[b], kind="coast", source=source))
    return out


def _segment_at(cfg, lat):
    return next((s["name"] for s in cfg["coast"]["segments"] if s["south"] <= lat <= s["north"]), None)


# --- islands --------------------------------------------------------------------

def _circle(isl, group, default_kind, source):
    """A named island as its placeholder circle."""
    return Landmass(isl["name"], group, isl["height_m"], [circle_ring(*isl["ref"], isl["radius_km"])],
                    kind=isl["kind"] or default_kind, source=source)


def _merge(rings, merge_km):
    """Label polygons by connected shorelines (within ``merge_km``); also a KD-tree of their samples.

    Returns ``(labels, tree, owner)``: the cluster of each ring, and a tree of
    points sampled along every ring with ``owner`` the ring of each point.
    Shorelines whose samples come within ``merge_km`` are connected, so the
    rule is exact to about the sampling step.
    """
    step = SAMPLE_KM if merge_km <= 0 else min(SAMPLE_KM, merge_km / 2)
    samples = [_sample(r, step) for r in rings]
    owner = np.repeat(np.arange(len(rings)), [len(s) for s in samples])
    tree = cKDTree(_xyz(np.vstack(samples)))
    if merge_km <= 0 or len(rings) < 2:
        return np.arange(len(rings)), tree, owner
    pairs = tree.query_pairs(float(geo.km_to_chord(merge_km)), output_type="ndarray")
    a, b = owner[pairs[:, 0]], owner[pairs[:, 1]]
    a, b = a[a != b], b[a != b]
    graph = coo_matrix((np.ones(len(a), dtype=np.int8), (a, b)), shape=(len(rings), len(rings)))
    return connected_components(graph, directed=False)[1], tree, owner


def _locate(ref, rings, bboxes, tree, owner, match_km):
    """Ring containing ``ref``, else the ring nearest it within ``match_km``: (ring, km) or (None, km)."""
    if not rings:
        return None, np.inf
    lon, lat = ref
    for p in np.flatnonzero(_in_box(lon, lat, bboxes.T)):
        if _inside(rings[p], lon, lat):
            return int(p), 0.0
    chord, j = tree.query(_xyz(ref)[0])
    km = float(geo.chord_to_km(chord))
    return (int(owner[j]), km) if km <= match_km else (None, km)


def _pool_landmasses(rings, merge_km, islands, group_of, default, match_km, rows, counts):
    """Merge a pool's polygons, match its named islands and make its landmasses.

    ``group_of(lon, lat)`` gives the group of a landmass from its centroid (the
    box group for a group's pool; the coast segment for coastal islands);
    ``default`` holds the group's kind, height_m and source for unnamed
    landmasses. A named island with no polygon within ``match_km`` becomes its
    placeholder circle. Appends one row per named island to ``rows``. Returns
    ``{group: [Landmass]}``: named landmasses in config order, then the rest
    by decreasing area.
    """
    out = defaultdict(list)
    clusters, geom, stats, labels, tree, owner, bboxes = [], [], [], None, None, None, None
    if rings:
        labels, tree, owner = _merge(rings, merge_km)
        bboxes = np.array([[r[:, 0].min(), r[:, 0].max(), r[:, 1].min(), r[:, 1].max()] for r in rings])
        clusters = [np.flatnonzero(labels == c) for c in range(labels.max() + 1)]
        geom = [[rings[p] for p in members] for members in clusters]
        stats = [_area_centroid(g) for g in geom]
        counts["merged"] = counts.get("merged", 0) + len(rings) - len(clusters)

    found = [_locate(isl["ref"], rings, bboxes, tree, owner, match_km) for isl in islands]
    named = defaultdict(list)   # cluster -> [(island, km from its ref)], in config order
    for isl, (ring, km) in zip(islands, found):
        if ring is not None:
            named[int(labels[ring])].append((isl, km))

    for isl, (ring, km) in zip(islands, found):
        if ring is None:
            log.warning("named island %s: no shoreline within %.0f km of its reference; "
                        "using its placeholder circle", isl["name"], match_km)
            counts["named_not_found"] = counts.get("named_not_found", 0) + 1
            group = group_of(*isl["ref"])
            rows.append(_row(isl, group, isl["name"], None, False, km, False))
            out[group].append(_circle(isl, group, default["kind"], "placeholder circle; " + isl["source"]))
            continue
        c = int(labels[ring])
        members = named[c]
        if members[0][0] is not isl:   # made with the first named island it holds
            continue
        _, lon, lat = stats[c]
        shifter = next((m for m, _ in members if m["shift_to_ref"]), None)
        if shifter is not None:
            geom[c] = [r + (np.array(shifter["ref"]) - (lon, lat)) for r in geom[c]]
        group = group_of(lon, lat)
        for m, m_km in members:
            rows.append(_row(m, group, isl["name"], (lon, lat), m_km == 0.0, m_km, shifter is not None))
        if len(members) > 1:
            log.info("named islands %s share one landmass, named %s", [m["name"] for m, _ in members], isl["name"])
        out[group].append(Landmass(isl["name"], group, max(m["height_m"] for m, _ in members), geom[c],
                                   kind=isl["kind"] or default["kind"], source=isl["source"]))

    count = defaultdict(int)
    for c in sorted(set(range(len(clusters))) - set(named), key=lambda c: -stats[c][0]):
        group = group_of(*stats[c][1:])
        if group is None:
            continue
        count[group] += 1
        out[group].append(Landmass(f"{group} #{count[group]}", group, default["height_m"], geom[c],
                                   kind=default["kind"], source=default["source"]))
    return out


def _row(isl, group, landmass, centroid, inside, nearest_km, shifted):
    c_lon, c_lat = centroid if centroid is not None else (np.nan, np.nan)
    return {"name": isl["name"], "group": group, "landmass": landmass, "found": centroid is not None,
            "inside": bool(inside), "nearest_km": np.nan if nearest_km is None else nearest_km,
            "centroid_km": float(geo.haversine_km(*isl["ref"], c_lon, c_lat)),
            "shifted": bool(shifted), "ref_type": isl["ref_type"], "ref_lon": isl["ref"][0],
            "ref_lat": isl["ref"][1], "centroid_lon": c_lon, "centroid_lat": c_lat}


# --- building -----------------------------------------------------------------

def _read_gshhg(path, domain, drop_ids):
    """Closed rings of the polygons whose bounding box meets the domain, and their GSHHG areas (km2).

    Polygons whose id is in ``drop_ids`` are left out; the third value returned is how many were.
    """
    try:
        import shapefile
    except ImportError as err:
        raise ImportError("reading GSHHG needs pyshp: pip install 'voyage-sim[fetch]'") from err
    lon_min, lon_max, lat_min, lat_max = domain
    rings, areas, dropped = [], [], set()
    with shapefile.Reader(str(path)) as reader:
        for sr in reader.iterShapeRecords(bbox=(lon_min, lat_min, lon_max, lat_max)):
            if sr.shape is None or not sr.shape.points:
                continue
            if (pid := str(sr.record["id"]).strip()) in drop_ids:
                dropped.add(pid)
                continue
            points = np.asarray(sr.shape.points, dtype=float)[:, :2]
            for part in np.split(points, list(sr.shape.parts)[1:]):
                rings.append(_closed(part))
                areas.append(float(sr.record["area"]))
    if missing := drop_ids - dropped:
        log.warning("drop_ids %s: no such polygon in the domain", sorted(missing))
    return rings, np.array(areas), len(dropped)


def _assemble(cfg, by_group, named_rows, counts):
    """Order landmasses (island groups in config order, then coast segments north to south)."""
    order = [(g["name"], g["region"]) for g in cfg["groups"]]
    order += [(s["name"], s["region"]) for s in cfg["coast"]["segments"]]
    landmasses, group_region = [], {}
    for name, region in order:
        if by_group.get(name):
            landmasses += by_group[name]
            group_region[name] = region
        else:
            log.warning("group %s has no land in the catalog", name)
    for kind in LANDMASS_KINDS:
        counts[f"landmasses_{kind}"] = sum(lm.kind == kind for lm in landmasses)
    named = pd.DataFrame(named_rows, columns=_NAMED_COLUMNS)
    return Catalog(landmasses, group_region, named, counts)


def _placeholder(cfg):
    """Named islands as circles and the coast from the config's polylines."""
    by_group = defaultdict(list)
    for g in cfg["groups"]:
        by_group[g["name"]] = [_circle(isl, g["name"], g["kind"], isl["source"]) for isl in g["islands"]]
    by_group.update(_coast_pieces(cfg["coast"]["placeholder"], cfg))
    for isl in cfg["coast"]["islands"]:
        seg = _segment_at(cfg, isl["ref"][1])
        by_group[seg].append(_circle(isl, seg, "island", isl["source"]))
    return _assemble(cfg, by_group, [], {})


def _from_gshhg(cfg, shp):
    """The catalog from GSHHG polygons (see the module docstring)."""
    coast, domain, mask = cfg["coast"], cfg["domain"], cfg["pacific_mask"]
    rings, areas, dropped = _read_gshhg(shp, domain, cfg["drop_ids"])
    continental = areas >= cfg["continent_min_area_km2"]
    counts = {"polygons": len(rings) + dropped, "dropped_by_id": dropped, "continental": int(continental.sum())}

    lines = []
    for p in np.flatnonzero(continental):
        ring = rings[p]
        keep = _in_box(ring[:, 0], ring[:, 1], domain) & _inside(mask, ring[:, 0], ring[:, 1])
        if keep.any():
            lines += _runs(ring, keep)
    coast_tree = cKDTree(_xyz(np.vstack(lines))) if lines else None

    islands = [rings[p] for p in np.flatnonzero(~continental)]
    cent = np.array([_area_centroid([r])[1:] for r in islands]).reshape(-1, 2)
    lon, lat = cent[:, 0], cent[:, 1]
    in_domain = _in_box(lon, lat, domain)
    group_idx = np.full(len(islands), -1)
    for k, g in enumerate(cfg["groups"]):
        for box in g["boxes"]:
            group_idx[(group_idx < 0) & in_domain & _in_box(lon, lat, box)] = k
    rest = in_domain & (group_idx < 0)
    pacific = rest & _inside(mask, lon, lat)

    # Distance of each remaining Pacific island's shoreline to the continental coast: nearest and farthest vertex.
    cand = np.flatnonzero(pacific)
    near, far = np.full(len(cand), np.inf), np.full(len(cand), np.inf)
    if coast_tree is not None and len(cand):
        verts = np.vstack([islands[i] for i in cand])
        starts = np.r_[0, np.cumsum([len(islands[i]) for i in cand])[:-1]]
        reach = float(geo.km_to_chord(coast["island_join_km"]))
        chord, _ = coast_tree.query(_xyz(verts), distance_upper_bound=reach)
        km = np.where(np.isfinite(chord), geo.chord_to_km(chord), np.inf)
        near, far = np.minimum.reduceat(km, starts), np.maximum.reduceat(km, starts)
    islet = far <= coast["islet_drop_km"]
    joins = ~islet & (near <= coast["island_join_km"])
    unassigned = cand[~islet & ~joins]
    counts.update(outside_domain=int((~in_domain).sum()), outside_mask=int((rest & ~pacific).sum()),
                  coastal_islets_dropped=int(islet.sum()), unassigned_dropped=len(unassigned),
                  in_groups=int((group_idx >= 0).sum()), coastal_islands=int(joins.sum()))
    for i in sorted(unassigned, key=lambda i: -_area_centroid([islands[i]])[0])[:20]:
        log.info("dropped polygon outside every group box, %.0f km from the coast: centroid (%.3f, %.3f)",
                 near[np.searchsorted(cand, i)], lon[i], lat[i])

    by_group, rows = defaultdict(list), []
    pools = [([islands[i] for i in np.flatnonzero(group_idx == k)], g["merge_km"], g["islands"],
              lambda lon, lat, name=g["name"]: name, g) for k, g in enumerate(cfg["groups"])]
    pools.append(([islands[i] for i in cand[joins]], cfg["merge_km"], coast["islands"],
                  lambda lon, lat: _segment_at(cfg, lat),
                  {"kind": "island", "height_m": coast["island_height_m"], "source": coast["island_source"]}))
    for pool, merge_km, named, group_of, default in pools:
        for name, lms in _pool_landmasses(pool, merge_km, named, group_of, default, cfg["match_km"], rows,
                                          counts).items():
            by_group[name] += lms
    for seg, pieces in _coast_pieces(lines, cfg).items():
        by_group[seg] = pieces + by_group[seg]
    catalog = _assemble(cfg, by_group, rows, counts)
    log.info("catalog: %s", catalog.counts)
    return catalog
