"""Is land in sight? Shorelines, sighting radii and the swept-segment test.

A voyage makes landfall under a sighting rule when it first comes within that
rule's radius of a shoreline. The radius is either one number for every
landmass (the 20/40/60 km sensitivity runs) or, per landmass, the distance at
which a target of its height clears the horizon for a lookout at a given eye
height, with standard refraction; islands and atolls (not coasts) are also
betrayed by their seabirds out to a fixed bird range, whichever is larger.

Distances to thousands of kilometres of coastline are too slow to compute for
every voyage at every step, so each rule gets a ``SightingRaster``: the
sighting margin (distance to the nearest shoreline minus that shoreline's
radius; negative means land in sight) on a lon/lat grid, built once and cached.
``LandfallDetector`` samples the rasters along each step's segment, so a
voyage that passes an atoll between two steps still sees it.

Geometry: a landmass is a list of rings, each a (k, 2) array of [lon, lat]
vertices joined in order. A closed shoreline repeats its first vertex at the
end; an open one (a stretch of continental coast) does not; a (1, 2) ring is a
single point.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
import yaml
from scipy.spatial import cKDTree

from voyage_sim import geo

LANDMASS_KINDS = ("island", "atoll", "coast")
SIGHTING_MODES = ("height", "uniform")
_KM_PER_DEG = np.radians(geo.EARTH_RADIUS_KM)  # great-circle km per degree of arc


# --- landmasses ---------------------------------------------------------------

def _parse_gshhg(spec):
    """Validate a GSHHG selector and normalize it to tuples of floats."""
    if not isinstance(spec, Mapping) or not spec or set(spec) - {"match_point", "box"}:
        raise ValueError("gshhg must be a mapping with match_point and/or box")
    out = {}
    if "match_point" in spec:
        point = np.asarray(spec["match_point"], dtype=float)
        if point.shape != (2,) or not np.all(np.isfinite(point)):
            raise ValueError("gshhg match_point must be [lon, lat]")
        out["match_point"] = (float(geo.wrap_lon(point[0])), float(point[1]))
    if "box" in spec:
        box = np.asarray(spec["box"], dtype=float)
        if box.shape != (4,) or not np.all(np.isfinite(box)) or box[0] >= box[1] or box[2] >= box[3]:
            raise ValueError("gshhg box must be [lon_min, lon_max, lat_min, lat_max] with min < max")
        out["box"] = tuple(float(b) for b in box)
    return out


@dataclass
class Landmass:
    """A sighting target: an island, an atoll, or a stretch of continental coast."""

    name: str
    group: str                 # archipelago or coast segment, e.g. "Marquesas", "Coast: Ecuador"
    height_m: float            # effective sighting height (atolls: vegetation height)
    rings: list[np.ndarray]
    kind: str = "island"
    gshhg: Mapping | None = None   # selector for attach_gshhg: match_point and/or box
    source: str = ""               # citation for the height

    def __post_init__(self):
        if self.kind not in LANDMASS_KINDS:
            raise ValueError(f"landmass {self.name!r}: kind must be one of {LANDMASS_KINDS}, got {self.kind!r}")
        self.height_m = float(self.height_m)
        if not 0 <= self.height_m < np.inf:
            raise ValueError(f"landmass {self.name!r}: height_m must be finite and non-negative")
        rings = []
        for ring in self.rings:
            ring = np.atleast_2d(np.array(ring, dtype=float))
            if (ring.ndim != 2 or ring.shape[1] != 2 or len(ring) == 0 or not np.all(np.isfinite(ring))
                    or np.any(np.abs(ring[:, 1]) > 90)):
                raise ValueError(f"landmass {self.name!r}: each ring must be a non-empty (k, 2) [lon, lat] array")
            ring[:, 0] = geo.wrap_lon(ring[:, 0])
            rings.append(ring)
        if not rings:
            raise ValueError(f"landmass {self.name!r}: needs at least one ring")
        self.rings = rings
        if self.gshhg is not None:
            self.gshhg = _parse_gshhg(self.gshhg)


def circle_ring(lon, lat, radius_km, n=64) -> np.ndarray:
    """Closed ring of ``n`` vertices (the first repeated at the end) ``radius_km`` around a point.

    Vertices are at the exact great-circle distance; a placeholder shoreline
    for islands not yet matched to GSHHG, and test geometry.
    """
    if n < 3 or not radius_km > 0:
        raise ValueError("circle_ring needs n >= 3 and a positive radius")
    bearing = np.radians(np.arange(n) * (360.0 / n))
    delta = radius_km / geo.EARTH_RADIUS_KM
    phi, lam = np.radians(lat), np.radians(lon)
    phi2 = np.arcsin(np.sin(phi) * np.cos(delta) + np.cos(phi) * np.sin(delta) * np.cos(bearing))
    lam2 = lam + np.arctan2(np.sin(bearing) * np.sin(delta) * np.cos(phi),
                            np.cos(delta) - np.sin(phi) * np.sin(phi2))
    ring = np.column_stack([geo.wrap_lon(np.degrees(lam2)), np.degrees(phi2)])
    return np.vstack([ring, ring[:1]])


# --- sighting rules -----------------------------------------------------------

def geometric_range_km(target_height_m, eye_height_m=3.0, refraction_k=0.13):
    """Distance (km) at which the top of a target just clears the horizon for a lookout.

    The target's and the eye's horizon distances add, each ``sqrt(2 R' h)`` on
    an Earth of effective radius ``R' = R / (1 - k)``; k = 0.13 is standard
    terrestrial refraction.
    """
    r_eff = geo.EARTH_RADIUS_KM / (1.0 - refraction_k)
    h_target = np.asarray(target_height_m, dtype=float) / 1000.0
    h_eye = np.asarray(eye_height_m, dtype=float) / 1000.0
    return np.sqrt(2.0 * r_eff * h_target) + np.sqrt(2.0 * r_eff * h_eye)


@dataclass(frozen=True)
class SightingRule:
    """How far off land counts as sighted: one radius for all, or by landmass height."""

    name: str
    mode: str = "height"       # "height" | "uniform"
    radius_km: float = 40.0    # uniform mode
    eye_height_m: float = 3.0
    refraction_k: float = 0.13
    scale: float = 1.0         # multiplies height-based radii, bird range included (sensitivity)
    bird_km: float = 0.0       # height mode: islands and atolls are in sight at least this far off
    floor_km: float = 0.0
    cap_km: float = np.inf

    def __post_init__(self):
        if self.mode not in SIGHTING_MODES:
            raise ValueError(f"sighting rule {self.name!r}: mode must be one of {SIGHTING_MODES}")
        if not (0 < self.radius_km < np.inf and 0 <= self.eye_height_m < np.inf and 0 <= self.refraction_k < 1
                and 0 < self.scale < np.inf and 0 <= self.bird_km < np.inf and 0 <= self.floor_km <= self.cap_km):
            raise ValueError(f"sighting rule {self.name!r}: need radius_km > 0, eye_height_m >= 0, "
                             "0 <= refraction_k < 1, scale > 0, bird_km >= 0 and 0 <= floor_km <= cap_km")

    def radius_km_for(self, height_m, kind) -> np.ndarray:
        """Sighting radius (km) for landmasses of the given heights (m) and kinds.

        Uniform mode: ``radius_km`` for every landmass. Height mode: the visual
        range, or for islands and atolls ``bird_km`` if larger (seabirds fishing
        off a coast say nothing about where it is), times ``scale``, then
        clipped to [floor_km, cap_km].
        """
        height, kind = np.broadcast_arrays(np.asarray(height_m, dtype=float), np.asarray(kind, dtype=str))
        if not np.isin(kind, LANDMASS_KINDS).all():
            raise ValueError(f"landmass kinds must be among {LANDMASS_KINDS}")
        if self.mode == "uniform":
            return np.full(height.shape, float(self.radius_km))
        visual = geometric_range_km(height, self.eye_height_m, self.refraction_k)
        reach = np.maximum(visual, np.where(kind == "coast", 0.0, self.bird_km))
        return np.clip(self.scale * reach, self.floor_km, self.cap_km)


# --- rasters ------------------------------------------------------------------

def _densify(ring, max_km):
    """Insert vertices, linearly in lon/lat, so consecutive ones are at most ``max_km`` apart."""
    if len(ring) < 2:
        return ring
    a, b = ring[:-1], ring[1:]
    seg_km = geo.haversine_km(a[:, 0], a[:, 1], b[:, 0], b[:, 1])
    pieces = np.maximum(np.ceil(seg_km / max_km), 1).astype(np.int64)
    seg = np.repeat(np.arange(len(a)), pieces)
    t = (np.arange(pieces.sum()) - np.repeat(np.cumsum(pieces) - pieces, pieces)) / pieces[seg]
    step = np.column_stack([geo.angle_diff(b[:, 0], a[:, 0]), b[:, 1] - a[:, 1]])
    pts = a[seg] + t[:, None] * step[seg]
    pts[:, 0] = geo.wrap_lon(pts[:, 0])
    return np.vstack([pts, ring[-1:]])


def _grid_shape(bounds, res_deg):
    """(ny, nx) cells tiling the bounds; a partial last cell still gets a row or column."""
    lon_min, lon_max, lat_min, lat_max = bounds
    nx = max(1, int(np.ceil((lon_max - lon_min) / res_deg - 1e-9)))
    ny = max(1, int(np.ceil((lat_max - lat_min) / res_deg - 1e-9)))
    return ny, nx


@dataclass(eq=False)
class SightingRaster:
    """Sighting margin on a lon/lat grid for one rule; negative where land is in sight.

    ``margin_km[j, i]``, at the centre of the cell in row j (from ``lat_min``)
    and column i (from ``lon_min``), is the minimum over landmasses of the
    great-circle distance to its shoreline minus its sighting radius, capped at
    ``cap_km``. ``index[j, i]`` is the landmass attaining it, -1 if none is
    within the cap. Cells inside a polygon get no special treatment: they lie
    inside the sighting zone anyway.
    """

    margin_km: np.ndarray
    index: np.ndarray
    bounds: tuple[float, float, float, float]   # (lon_min, lon_max, lat_min, lat_max)
    res_deg: float
    cap_km: float
    rule_name: str = ""
    radius_km: np.ndarray = field(default_factory=lambda: np.zeros(0))  # per landmass
    landmass_names: tuple[str, ...] = ()

    def __post_init__(self):
        self.bounds = tuple(float(b) for b in self.bounds)
        self.res_deg = float(self.res_deg)
        self.cap_km = float(self.cap_km)
        self.margin_km = np.asarray(self.margin_km, dtype=np.float32)
        self.index = np.asarray(self.index)
        self.radius_km = np.asarray(self.radius_km, dtype=float)
        self.landmass_names = tuple(str(s) for s in self.landmass_names)
        shape = _grid_shape(self.bounds, self.res_deg)
        if self.margin_km.shape != shape or self.index.shape != shape:
            raise ValueError(f"raster arrays must be {shape} for bounds {self.bounds} at {self.res_deg} deg")
        if not np.issubdtype(self.index.dtype, np.integer):
            raise ValueError("raster index must be an integer array")

    @property
    def cell_km(self) -> float:
        """Cell size in km along its larger axis (north-south)."""
        return self.res_deg * _KM_PER_DEG

    @classmethod
    def build(cls, landmasses, rule, bounds, res_deg=0.05, densify_km=1.0, cap_km=250.0) -> SightingRaster:
        """Rasterize the sighting margin of ``landmasses`` under ``rule`` over ``bounds``.

        Distances are exact great-circle distances from cell centres to
        shoreline vertices densified to <= ``densify_km`` (a KD-tree on unit
        vectors per landmass), so they overestimate the distance to the
        shoreline itself by at most ``densify_km / 2``. Each landmass is
        evaluated only on cells within its radius plus ``cap_km``.
        """
        lon_min, lon_max, lat_min, lat_max = bounds = tuple(float(b) for b in bounds)
        if not (-180 <= lon_min < lon_max <= 180 and -90 <= lat_min < lat_max <= 90
                and res_deg > 0 and densify_km > 0 and cap_km > 0):
            # Landmass longitudes are wrapped to [-180, 180): bounds in 0..360 would silently miss every one.
            raise ValueError("need -180 <= lon_min < lon_max <= 180, -90 <= lat_min < lat_max <= 90 "
                             "and positive res_deg, densify_km, cap_km")
        landmasses = list(landmasses)
        ny, nx = _grid_shape(bounds, res_deg)
        lon_c = lon_min + (np.arange(nx) + 0.5) * res_deg
        lat_c = lat_min + (np.arange(ny) + 0.5) * res_deg
        radii = rule.radius_km_for([lm.height_m for lm in landmasses], [lm.kind for lm in landmasses])
        margin = np.full((ny, nx), float(cap_km))
        index = np.full((ny, nx), -1, dtype=np.int16 if len(landmasses) < 2**15 else np.int32)
        for k, (lm, radius) in enumerate(zip(landmasses, radii)):
            verts = np.vstack([_densify(ring, densify_km) for ring in lm.rings])
            reach = radius + cap_km
            dlat = reach / _KM_PER_DEG
            # Taking cos(lat) at the most poleward latitude in reach keeps the box conservative.
            dlon = dlat / np.cos(np.radians(min(np.abs(verts[:, 1]).max() + dlat, 89.0)))
            i0 = np.searchsorted(lon_c, verts[:, 0].min() - dlon, side="left")
            i1 = np.searchsorted(lon_c, verts[:, 0].max() + dlon, side="right")
            j0 = np.searchsorted(lat_c, verts[:, 1].min() - dlat, side="left")
            j1 = np.searchsorted(lat_c, verts[:, 1].max() + dlat, side="right")
            if i0 >= i1 or j0 >= j1:
                continue
            glon, glat = np.meshgrid(lon_c[i0:i1], lat_c[j0:j1])
            tree = cKDTree(geo.lonlat_to_xyz(verts[:, 0], verts[:, 1]))
            chord, _ = tree.query(geo.lonlat_to_xyz(glon, glat),
                                  distance_upper_bound=float(geo.km_to_chord(reach)), workers=-1)
            sub = np.where(np.isfinite(chord), geo.chord_to_km(chord) - radius, np.inf)
            view_m, view_i = margin[j0:j1, i0:i1], index[j0:j1, i0:i1]
            better = sub < view_m
            view_m[better] = sub[better]
            view_i[better] = k
        return cls(margin, index, bounds, res_deg, cap_km, rule_name=rule.name, radius_km=radii,
                   landmass_names=tuple(lm.name for lm in landmasses))

    def query(self, lon, lat) -> tuple[np.ndarray, np.ndarray]:
        """Margin (km, bilinear on cell centres) and landmass index (nearest cell) at points.

        Points outside the bounds, or not finite, get ``(cap_km, -1)``; within
        half a cell of the bounds the edge cells' values extend outwards.
        """
        lon = np.asarray(lon, dtype=float)
        lat = np.asarray(lat, dtype=float)
        lon_min, lon_max, lat_min, lat_max = self.bounds
        ny, nx = self.margin_km.shape
        inside = (lon >= lon_min) & (lon <= lon_max) & (lat >= lat_min) & (lat <= lat_max)
        fx = np.clip((np.where(inside, lon, lon_min) - lon_min) / self.res_deg - 0.5, 0, nx - 1)
        fy = np.clip((np.where(inside, lat, lat_min) - lat_min) / self.res_deg - 0.5, 0, ny - 1)
        i0 = np.minimum(fx.astype(np.int64), max(nx - 2, 0))
        j0 = np.minimum(fy.astype(np.int64), max(ny - 2, 0))
        i1, j1 = np.minimum(i0 + 1, nx - 1), np.minimum(j0 + 1, ny - 1)
        a, b = fx - i0, fy - j0
        m = self.margin_km
        margin = ((1 - a) * (1 - b) * m[j0, i0] + a * (1 - b) * m[j0, i1]
                  + (1 - a) * b * m[j1, i0] + a * b * m[j1, i1])
        index = self.index[np.rint(fy).astype(np.int64), np.rint(fx).astype(np.int64)]
        return np.where(inside, margin, self.cap_km), np.where(inside, index, -1)

    def save(self, path):
        """Write an ``.npz`` cache at exactly ``path``."""
        with open(path, "wb") as fh:
            np.savez_compressed(fh, margin_km=self.margin_km, index=self.index, bounds=np.array(self.bounds),
                                res_deg=self.res_deg, cap_km=self.cap_km, rule_name=np.array(self.rule_name),
                                radius_km=self.radius_km, landmass_names=np.array(self.landmass_names, dtype=str))

    @classmethod
    def load(cls, path) -> SightingRaster:
        """Read a raster written by ``save``."""
        with np.load(path, allow_pickle=False) as z:
            return cls(z["margin_km"], z["index"], tuple(z["bounds"]), float(z["res_deg"]), float(z["cap_km"]),
                       rule_name=str(z["rule_name"]), radius_km=z["radius_km"],
                       landmass_names=tuple(z["landmass_names"]))


# --- detection ----------------------------------------------------------------

class LandfallDetector:
    """First sighting of land along each voyage's step, under every sighting rule at once."""

    def __init__(self, landmasses, rasters, substep_km=2.0):
        self.landmasses = list(landmasses)
        self.rasters = list(rasters)
        if not self.rasters:
            raise ValueError("LandfallDetector needs at least one raster")
        if not 0 < substep_km < np.inf:
            raise ValueError("substep_km must be positive")
        self.substep_km = float(substep_km)
        self.landmass_names = [lm.name for lm in self.landmasses]
        self.rule_names = [r.rule_name for r in self.rasters]
        self.groups = list(dict.fromkeys(lm.group for lm in self.landmasses))
        lookup = {g: i for i, g in enumerate(self.groups)}
        self.landmass_group = np.array([lookup[lm.group] for lm in self.landmasses], dtype=np.int64)
        # Group of a raster index; the trailing -1 maps index -1 (no landmass) to group -1.
        self._group_of = np.append(self.landmass_group, -1)
        for r in self.rasters:
            if r.landmass_names and list(r.landmass_names) != self.landmass_names:
                raise ValueError(f"raster {r.rule_name!r} was built for different landmasses")
            if r.index.size and r.index.max() >= len(self.landmasses):
                raise ValueError(f"raster {r.rule_name!r} refers to more than {len(self.landmasses)} landmasses")

    def check_segment(self, lon0, lat0, lon1, lat1, origin_group, armed, active, eligible=None, rules=None,
                      return_exact=False):
        """Where along each step's segment land first comes into sight, per rule.

        The segment runs linearly in lon/lat from (lon0, lat0) to (lon1, lat1).
        ``origin_group`` is each voyage's home group index (-1 for none), or an
        (n, n_groups) bool mask of its home groups.
        ``eligible``, optional (n, n_groups) bool, marks the groups each voyage
        may land on (its destinations); land of other groups is sailed past,
        though it still keeps a voyage from being out of sight of all land.
        ``rules`` lists the raster indices to check (default all, in order);
        ``armed`` and ``active`` are bool, shape (n,) or (len(rules), n).

        Returns ``(hit, frac, landmass, armed_new)``, each (len(rules), n):
        whether land that counts came into sight, where along the segment (frac
        in [0, 1], NaN if not), which landmass (-1 if not), and the updated
        armed flags. With ``return_exact``, also ``exact``: False where the
        sighting was placed at a switch from land in sight that does not count
        (see the limitation below), so it may have come into sight earlier.

        A segment is skipped, as wholly out of sight, when
        ``min(margin_start, margin_end) > 0.75 * length + cell_km``. That is
        safe: the true margin changes by at most 1 km per km travelled and its
        bilinear interpolant by at most sqrt(2), so no point of the segment lies
        more than ``sqrt(2) / 2 * length`` below the smaller end margin; the
        cell term absorbs the rest. Otherwise the margin is sampled every
        ``substep_km`` including both ends, and the first sighting is located by
        linear interpolation of the margin between the samples around it.

        Home exclusion: a voyage starts in sight of its home islands, so
        sightings of its home groups count only once it has been out of sight
        of all land (margin > 0) at some point, in an earlier step or earlier in
        this segment. Inactive voyages, and segments with a non-finite end, get
        no sighting and keep their armed flag.

        Limitation: a raster keeps only the landmass of smallest margin per
        cell, so where an eligible landmass's sighting zone overlaps that of an
        ineligible one with a smaller margin there (or of a home landmass not
        yet armed), the eligible one is sighted only where its own margin takes
        over, and that switch is not interpolated (``exact`` is False). For
        destinations this matters little in the study area: the catalog joins
        small islands just off the American coast (Gorgona, La Plata, Isla
        Mocha, ...) to their coast segment, and eastern Polynesian groups lie
        hundreds of km from non-destination land. It does matter for other land
        sighted near home, whose zone overlaps the home zone: a voyage leaving
        the South Marquesas sights the North Marquesas (Ua Pou) only where their
        margin takes over from its unarmed home's, often days late and tens of
        km inside their range, and the same holds for a coast segment next to a
        departure port.
        """
        lon0, lat0, lon1, lat1 = np.broadcast_arrays(
            *(np.asarray(x, dtype=float) for x in (lon0, lat0, lon1, lat1)))
        n = lon0.shape[0]
        origin_group = np.asarray(origin_group)
        home = None
        if origin_group.dtype == bool:
            home = self._group_mask(origin_group, n, "origin_group")
        else:
            origin_group = np.broadcast_to(origin_group, lon0.shape)
        rasters = list(self.rasters) if rules is None else [self.rasters[r] for r in rules]
        if eligible is not None:
            eligible = self._group_mask(eligible, n, "eligible")
        shape = (len(rasters), n)
        armed_new = np.broadcast_to(np.asarray(armed, dtype=bool), shape).copy()
        finite = np.isfinite(lon0) & np.isfinite(lat0) & np.isfinite(lon1) & np.isfinite(lat1)
        active = np.broadcast_to(np.asarray(active, dtype=bool), shape) & finite
        hit = np.zeros(shape, dtype=bool)
        frac = np.full(shape, np.nan)
        landmass = np.full(shape, -1, dtype=np.int64)
        exact = np.zeros(shape, dtype=bool)

        length = geo.haversine_km(lon0, lat0, lon1, lat1)
        dlon = geo.angle_diff(lon1, lon0)
        dlat = lat1 - lat0
        for r, raster in enumerate(rasters):
            m_start, _ = raster.query(lon0, lat0)
            m_end, _ = raster.query(lon1, lat1)
            far = np.minimum(m_start, m_end) > 0.75 * length + raster.cell_km
            armed_new[r] |= active[r] & far
            cand = np.flatnonzero(active[r] & ~far)
            if cand.size == 0:
                continue

            n_samples = np.ceil(length[cand] / self.substep_km).astype(np.int64) + 1
            # Padding repeats the end point, which changes neither the first sighting nor arming.
            t = np.minimum(np.arange(n_samples.max()) / np.maximum(n_samples - 1, 1)[:, None], 1.0)
            lon = geo.wrap_lon(lon0[cand, None] + t * dlon[cand, None])
            lat = lat0[cand, None] + t * dlat[cand, None]
            m, idx = raster.query(lon, lat)
            out_of_sight = m > 0
            armed_so_far = armed_new[r, cand][:, None] | np.logical_or.accumulate(out_of_sight, axis=1)
            group = self._group_of[idx]
            is_home = group == origin_group[cand, None] if home is None else home[cand[:, None], group]
            counts = ~out_of_sight & (idx >= 0) & (~is_home | armed_so_far)
            if eligible is not None:
                counts &= eligible[cand[:, None], group]

            rows = np.arange(cand.size)
            k = counts.argmax(axis=1)
            kp = np.maximum(k - 1, 0)
            m_prev, m_k = m[rows, kp], m[rows, k]
            # Interpolate the zero crossing when the previous sample was out of sight; otherwise
            # (first sample, or a switch from land that does not count) the sighting is at sample k.
            crossing = (k > 0) & (m_prev > 0)
            w = np.divide(m_prev, m_prev - m_k, out=np.ones_like(m_prev), where=crossing)
            f = t[rows, kp] + w * (t[rows, k] - t[rows, kp])

            seen = counts[rows, k]
            hit[r, cand] = seen
            frac[r, cand[seen]] = f[seen]
            landmass[r, cand[seen]] = idx[rows, k][seen]
            exact[r, cand[seen]] = (crossing | (k == 0))[seen]
            armed_new[r, cand] |= out_of_sight.any(axis=1)
        if return_exact:
            return hit, frac, landmass, armed_new, exact
        return hit, frac, landmass, armed_new

    def _group_mask(self, mask, n, what):
        """An (n, n_groups) bool mask, with a trailing False column for group -1 (no landmass)."""
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (n, len(self.groups)):
            raise ValueError(f"{what} must have shape (n, n_groups) = {(n, len(self.groups))}, got {mask.shape}")
        return np.hstack([mask, np.zeros((n, 1), dtype=bool)])


# --- catalog and shorelines ---------------------------------------------------

_GEOMETRY_KEYS = ("point", "polyline", "polygon")
_CATALOG_KEYS = {"name", "group", "kind", "height_m", "radius_km", "gshhg", "source", *_GEOMETRY_KEYS}


def _check_keys(cfg, allowed, required, what):
    if not isinstance(cfg, Mapping):
        raise ValueError(f"{what} must be a mapping")
    if unknown := set(cfg) - set(allowed):
        raise ValueError(f"unknown {what} keys {sorted(unknown)}; allowed: {sorted(allowed)}")
    if missing := set(required) - set(cfg):
        raise ValueError(f"missing {what} keys {sorted(missing)}")


def _landmass_from_config(cfg):
    _check_keys(cfg, _CATALOG_KEYS, {"name", "group", "height_m"}, "landmass")
    geometry = [k for k in _GEOMETRY_KEYS if k in cfg]
    if len(geometry) != 1:
        raise ValueError(f"give exactly one geometry, one of {_GEOMETRY_KEYS}")
    geometry = geometry[0]
    if "radius_km" in cfg and geometry != "point":
        raise ValueError("radius_km applies only to point geometry")
    coords = np.asarray(cfg[geometry], dtype=float)
    if geometry == "point":
        if coords.shape != (2,):
            raise ValueError("point must be [lon, lat]")
        ring = circle_ring(coords[0], coords[1], float(cfg["radius_km"])) if "radius_km" in cfg else coords[None, :]
    else:
        if coords.ndim != 2 or coords.shape[1] != 2 or len(coords) < (3 if geometry == "polygon" else 2):
            raise ValueError(f"{geometry} must be a list of [lon, lat] pairs (polygons at least 3)")
        ring = coords
        if geometry == "polygon" and not np.array_equal(ring[0], ring[-1]):
            ring = np.vstack([ring, ring[:1]])
    return Landmass(name=str(cfg["name"]), group=str(cfg["group"]), height_m=float(cfg["height_m"]), rings=[ring],
                    kind=str(cfg.get("kind", "island")), gshhg=cfg.get("gshhg"), source=str(cfg.get("source", "")))


def load_catalog(path) -> list[Landmass]:
    """Read a landmass catalog (``configs/landmasses.yaml``).

    Schema::

        landmasses:
          - name: Nuku Hiva          # unique; how results name the landmass
            group: Marquesas         # archipelago or coast segment; home exclusion and
                                     # destination odds work by group
            kind: island             # island | atoll | coast (default island)
            height_m: 1224           # effective sighting height (atolls: vegetation, ~25 m)
            source: "..."            # optional citation for the height
            # geometry, exactly one of:
            point: [-140.1, -8.87]   # with radius_km: a circle placeholder; without: one point
            radius_km: 9
            polyline: [[lon, lat], ...]   # open shoreline, e.g. a stretch of coast
            polygon: [[lon, lat], ...]    # closed shoreline (closed here if not already)
            # optional, used by attach_gshhg to swap in the real shoreline:
            gshhg: {match_point: [lon, lat], box: [lon_min, lon_max, lat_min, lat_max]}
    """
    with open(path, encoding="utf-8") as fh:
        doc = yaml.safe_load(fh)
    if not isinstance(doc, Mapping) or not isinstance(doc.get("landmasses"), list):
        raise ValueError(f"{path}: expected a top-level 'landmasses' list")
    landmasses = []
    for n, cfg in enumerate(doc["landmasses"]):
        try:
            landmasses.append(_landmass_from_config(cfg))
        except (TypeError, ValueError) as err:
            name = cfg.get("name", f"#{n}") if isinstance(cfg, Mapping) else f"#{n}"
            raise ValueError(f"{path}: landmass {name!r}: {err}") from err
    names = [lm.name for lm in landmasses]
    if dupes := sorted({s for s in names if names.count(s) > 1}):
        raise ValueError(f"{path}: duplicate landmass names {dupes}")
    return landmasses


def _contains(ring, lon, lat):
    """Even-odd point-in-polygon test for a closed ring."""
    x0, y0, x1, y1 = ring[:-1, 0], ring[:-1, 1], ring[1:, 0], ring[1:, 1]
    spans = (y0 > lat) != (y1 > lat)
    x_cross = x0[spans] + (lat - y0[spans]) * (x1[spans] - x0[spans]) / (y1[spans] - y0[spans])
    return np.count_nonzero(lon < x_cross) % 2 == 1


def _area(ring):
    """Shoelace area of a closed ring, square degrees."""
    x, y = ring[:, 0], ring[:, 1]
    return 0.5 * abs(np.dot(x[:-1], y[1:]) - np.dot(x[1:], y[:-1]))


def _runs_in_box(ring, box):
    """Contiguous runs of a ring's vertices inside a lon/lat box, as open polylines."""
    lon_min, lon_max, lat_min, lat_max = box
    inside = (ring[:, 0] >= lon_min) & (ring[:, 0] <= lon_max) & (ring[:, 1] >= lat_min) & (ring[:, 1] <= lat_max)
    if inside.all():
        return [ring]
    if len(ring) > 1 and np.array_equal(ring[0], ring[-1]):
        # Closed ring: start at an outside vertex so that no run wraps past the end.
        start = int(np.argmin(inside[:-1]))
        ring, inside = np.roll(ring[:-1], -start, axis=0), np.roll(inside[:-1], -start)
    edges = np.diff(np.concatenate([[0], inside.astype(np.int8), [0]]))
    return [ring[a:b] for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))]


def attach_gshhg(landmasses, shp_path) -> list[Landmass]:
    """Swap in GSHHG shorelines for landmasses that carry a ``gshhg`` selector.

    ``match_point`` selects the shoreline polygon containing that point (the
    smallest, if several do; for an atoll, put it on a motu, not in the
    lagoon). ``box`` keeps only vertices inside [lon_min, lon_max, lat_min,
    lat_max], as one open polyline per contiguous run, from the matched polygon
    or, without ``match_point``, from every polygon reaching into the box.
    Other landmasses are returned unchanged. The shapefile (lon in [-180, 180),
    as distributed) is read once, and only shapes near a selector are parsed.
    Needs pyshp.
    """
    try:
        import shapefile
    except ImportError as err:
        raise ImportError("attach_gshhg needs pyshp: pip install 'voyage-sim[fetch]'") from err

    landmasses = list(landmasses)
    todo = [i for i, lm in enumerate(landmasses) if lm.gshhg]
    if not todo:
        return landmasses
    # Search box per selector as (xmin, ymin, xmax, ymax): the match point itself, else the clip box.
    boxes = []
    for i in todo:
        spec = landmasses[i].gshhg
        if "match_point" in spec:
            lon, lat = spec["match_point"]
            boxes.append((lon, lat, lon, lat))
        else:
            lon_min, lon_max, lat_min, lat_max = spec["box"]
            boxes.append((lon_min, lat_min, lon_max, lat_max))
    boxes = np.array(boxes)

    polygons, near = [], [[] for _ in todo]
    with shapefile.Reader(str(shp_path)) as reader:
        union = (boxes[:, 0].min(), boxes[:, 1].min(), boxes[:, 2].max(), boxes[:, 3].max())
        for shape in reader.iterShapes(bbox=union):
            if shape is None or not shape.points:
                continue
            x0, y0, x1, y1 = shape.bbox
            hits = np.flatnonzero((x0 <= boxes[:, 2]) & (x1 >= boxes[:, 0])
                                  & (y0 <= boxes[:, 3]) & (y1 >= boxes[:, 1]))
            if hits.size == 0:
                continue
            points = np.asarray(shape.points, dtype=float)[:, :2]
            for part in np.split(points, list(shape.parts)[1:]):
                if not np.array_equal(part[0], part[-1]):
                    part = np.vstack([part, part[:1]])
                for s in hits:
                    near[s].append(len(polygons))
                polygons.append(part)

    for s, i in enumerate(todo):
        lm, spec = landmasses[i], landmasses[i].gshhg
        rings = [polygons[p] for p in near[s]]
        if "match_point" in spec:
            rings = [ring for ring in rings if _contains(ring, *spec["match_point"])]
            if not rings:
                raise ValueError(f"landmass {lm.name!r}: no polygon in {shp_path} contains "
                                 f"match_point {spec['match_point']}")
            rings = [min(rings, key=_area)]
        if "box" in spec:
            rings = [run for ring in rings for run in _runs_in_box(ring, spec["box"])]
            if not rings:
                raise ValueError(f"landmass {lm.name!r}: no shoreline vertices inside {spec['box']}")
        landmasses[i] = dataclasses.replace(lm, rings=rings)
    return landmasses
