"""Vessel performance: how a hull and rig move through the water.

A sailing vessel's speed comes from a polar: boat speed through the water by
true wind angle (TWA) and true wind speed (TWS), in knots. It sails only when
the wind allows: no closer to the wind than ``min_twa_deg``, not becalmed
(below ``min_sail_tws_kn``) and not hove to in a gale (at or above
``heave_to_tws_kn``). The rest of the time, and always for drift craft, it moves
by leeway, the wind's push on the hull: a downwind and a crosswind part, each
linear in wind speed (Allen & Plourde 1999).

Winds here are relative to the water (wind minus current); the simulator adds
the current afterwards. Vessel definitions, with a citation for every number,
live in ``configs/vessels.yaml``.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
import yaml

from voyage_sim import geo

VESSEL_KINDS = ("sailing", "drift")


# --- polars -------------------------------------------------------------------

def _bracket(grid, x):
    """Lower cell index and fractional position of ``x`` on an ascending grid, clipped to its ends."""
    x = np.clip(x, grid[0], grid[-1])
    i = np.clip(np.searchsorted(grid, x, side="right") - 1, 0, len(grid) - 2)
    return i, (x - grid[i]) / (grid[i + 1] - grid[i])


@dataclass(frozen=True)
class Polar:
    """Boat speed through the water (knots) tabulated by TWA (deg) and TWS (knots)."""

    twa_deg: np.ndarray
    tws_kn: np.ndarray
    speed_kn: np.ndarray

    def __post_init__(self):
        twa = np.array(self.twa_deg, dtype=float)
        tws = np.array(self.tws_kn, dtype=float)
        speed = np.array(self.speed_kn, dtype=float)
        if twa.ndim != 1 or len(twa) < 2 or not np.all(np.diff(twa) > 0) or twa[0] < 0 or twa[-1] > 180:
            raise ValueError("polar twa_deg must be at least 2 strictly ascending angles within [0, 180]")
        if tws.ndim != 1 or len(tws) < 2 or not np.all(np.diff(tws) > 0) or tws[0] != 0:
            raise ValueError("polar tws_kn must be at least 2 strictly ascending speeds starting at 0")
        if speed.shape != (len(twa), len(tws)) or not np.all(np.isfinite(speed)) or np.any(speed < 0):
            raise ValueError(f"polar speed_kn must be finite, non-negative, shape {(len(twa), len(tws))}")
        for name, arr in (("twa_deg", twa), ("tws_kn", tws), ("speed_kn", speed)):
            arr.flags.writeable = False
            object.__setattr__(self, name, arr)

    def __eq__(self, other):
        if not isinstance(other, Polar):
            return NotImplemented
        return all(np.array_equal(getattr(self, f), getattr(other, f)) for f in ("twa_deg", "tws_kn", "speed_kn"))

    def speed(self, twa_deg, tws_kn) -> np.ndarray:
        """Boat speed in knots: bilinear in the table, inputs clipped to its range."""
        i, a = _bracket(self.twa_deg, np.asarray(twa_deg, dtype=float))
        j, b = _bracket(self.tws_kn, np.asarray(tws_kn, dtype=float))
        s = self.speed_kn
        return ((1 - a) * (1 - b) * s[i, j] + a * (1 - b) * s[i + 1, j]
                + (1 - a) * b * s[i, j + 1] + a * b * s[i + 1, j + 1])


def parametric_polar(min_twa_deg, max_speed_kn, wind_ratio, angle_factors) -> Polar:
    """Polar for ``speed = min(max_speed_kn, wind_ratio * TWS) * f(TWA)``.

    ``f`` is piecewise-linear through ``angle_factors`` [(twa, factor), ...],
    flat beyond its first and last points, and zero below ``min_twa_deg``. The
    table has a knot at every kink, so its bilinear lookup reproduces the
    formula exactly.
    """
    pts = np.asarray(angle_factors, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) == 0:
        raise ValueError("angle_factors must be a non-empty list of [twa_deg, factor] pairs")
    a_twa, a_f = pts.T
    if not np.all(np.diff(a_twa) > 0) or not 0 <= a_twa[0] <= a_twa[-1] <= 180:
        raise ValueError("angle_factors angles must be strictly ascending within [0, 180]")
    if not np.all(np.isfinite(a_f)) or np.any(a_f < 0):
        raise ValueError("angle_factors factors must be finite and non-negative")
    if not 0 <= min_twa_deg <= 180:
        raise ValueError("min_twa_deg must be within [0, 180]")
    if not (0 < max_speed_kn < np.inf and 0 < wind_ratio < np.inf):
        raise ValueError("max_speed_kn and wind_ratio must be positive and finite")

    inner = a_twa[(a_twa > min_twa_deg) & (a_twa < 180.0)]
    twa = np.unique(np.concatenate([[min_twa_deg], inner, [180.0]]))
    f = np.interp(twa, a_twa, a_f)
    if min_twa_deg > 0:
        # A knot one ulp below min_twa makes the step to zero exact under bilinear lookup.
        twa = np.concatenate([[0.0, np.nextafter(min_twa_deg, 0.0)], twa])
        f = np.concatenate([[0.0, 0.0], f])
    # Speed is linear in TWS up to the cap and flat beyond; clipping supplies the flat part.
    return Polar(twa, np.array([0.0, max_speed_kn / wind_ratio]), np.outer(f, [0.0, max_speed_kn]))


# --- leeway -------------------------------------------------------------------

@dataclass(frozen=True)
class Leeway:
    """Wind drift through the water, Allen & Plourde (1999) form ``L = slope * W10 + offset``.

    Slopes are m/s of drift per m/s of 10 m wind; offsets are m/s.
    """

    dw_slope: float
    dw_offset_ms: float = 0.0
    cw_slope: float = 0.0
    cw_offset_ms: float = 0.0

    def __post_init__(self):
        for f in dataclasses.fields(self):
            value = getattr(self, f.name)
            if not np.isfinite(value) or (f.name.endswith("slope") and value < 0):
                raise ValueError(f"leeway {f.name} must be finite (slopes non-negative), got {value}")

    def velocity(self, wind_u, wind_v, cw_sign) -> tuple[np.ndarray, np.ndarray]:
        """Leeway (u, v) m/s for a wind relative to the water.

        The downwind part points along the wind vector; the crosswind part is
        perpendicular, to the right of downwind (clockwise) for ``cw_sign = +1``
        and to the left for -1. Each magnitude is floored at 0 before the sign
        is applied, and both are zero in calm air.
        """
        wind_u = np.asarray(wind_u, dtype=float)
        wind_v = np.asarray(wind_v, dtype=float)
        w = np.hypot(wind_u, wind_v)
        down = np.maximum(0.0, self.dw_slope * w + self.dw_offset_ms)
        cross = np.maximum(0.0, self.cw_slope * w + self.cw_offset_ms) * np.asarray(cw_sign, dtype=float)
        # Unit downwind vector (zero in calm air); its right-hand normal is (ev, -eu).
        safe_w = np.where(w > 0, w, 1.0)
        eu, ev = wind_u / safe_w, wind_v / safe_w
        return down * eu + cross * ev, down * ev - cross * eu


# --- vessels ------------------------------------------------------------------

@dataclass(frozen=True)
class Vessel:
    """A craft's sailing envelope, its polar, and its leeway when not sailing."""

    name: str
    kind: str                  # "sailing" | "drift"
    leeway: Leeway             # used whenever the vessel is not sailing
    polar: Polar | None = None
    min_twa_deg: float = 180.0       # closest angle to the wind it can sail
    min_sail_tws_kn: float = 0.0     # below this: becalmed, drifts
    heave_to_tws_kn: float = np.inf  # at or above this: stops sailing, drifts
    speed_scale: float = 1.0         # sensitivity knob on polar speeds
    sources: Mapping[str, str] = field(default_factory=dict)  # parameter -> citation

    def __post_init__(self):
        if self.kind not in VESSEL_KINDS:
            raise ValueError(f"vessel {self.name!r}: kind must be one of {VESSEL_KINDS}, got {self.kind!r}")
        if (self.kind == "sailing") != (self.polar is not None):
            raise ValueError(f"vessel {self.name!r}: a sailing vessel needs a polar and a drift vessel has none")
        if not 0 <= self.min_twa_deg <= 180:
            raise ValueError(f"vessel {self.name!r}: min_twa_deg must be within [0, 180]")
        if not 0 <= self.min_sail_tws_kn < self.heave_to_tws_kn:
            raise ValueError(f"vessel {self.name!r}: need 0 <= min_sail_tws_kn < heave_to_tws_kn")
        if not 0 < self.speed_scale < np.inf:
            raise ValueError(f"vessel {self.name!r}: speed_scale must be positive and finite")
        if self.polar is not None:
            # Wherever sailable says "sail", the polar must make way, or the voyage
            # would sit still with no leeway either. Speed is piecewise-linear in TWA
            # between table angles, so checking min_twa_deg and every angle above it
            # covers the whole sailable range. This catches a polar whose zero region
            # extends past min_twa_deg (e.g. after lowering min_twa_deg by override).
            angles = np.append(self.min_twa_deg, self.polar.twa_deg[self.polar.twa_deg > self.min_twa_deg])
            if np.any(self.polar.speed(angles, self.polar.tws_kn[-1]) <= 0):
                raise ValueError(f"vessel {self.name!r}: polar must give positive speed at every angle "
                                 f"from min_twa_deg ({self.min_twa_deg}) to 180")
        object.__setattr__(self, "sources", dict(self.sources))

    @property
    def can_sail(self) -> bool:
        """Whether the vessel ever sails (drift craft never do)."""
        return self.kind == "sailing"

    def sailable(self, twa_deg, tws_kn) -> np.ndarray:
        """Where the vessel can sail at this true wind angle (deg) and speed (knots)."""
        twa = np.asarray(twa_deg, dtype=float)
        tws = np.asarray(tws_kn, dtype=float)
        ok = (twa >= self.min_twa_deg) & (tws >= self.min_sail_tws_kn) & (tws < self.heave_to_tws_kn)
        return ok & self.can_sail

    def water_velocity(self, heading_deg, wind_u, wind_v, cw_sign, speed_factor=1.0):
        """Velocity through the water ``(u, v, sailing)``, m/s.

        ``wind_*`` is the wind relative to the water. Where the heading is finite
        and the wind is sailable, the vessel makes its polar speed (times
        ``speed_scale`` and the per-voyage ``speed_factor``) along the heading;
        everywhere else it drifts by leeway. ``sailing`` is that mask.
        """
        heading, wind_u, wind_v, cw_sign, speed_factor = np.broadcast_arrays(
            *(np.asarray(x, dtype=float) for x in (heading_deg, wind_u, wind_v, cw_sign, speed_factor)))
        drift_u, drift_v = self.leeway.velocity(wind_u, wind_v, cw_sign)
        tws_kn = np.hypot(wind_u, wind_v) / geo.KNOT_MS
        twa = geo.true_wind_angle(heading, geo.wind_from_deg(wind_u, wind_v))
        sailing = np.isfinite(heading) & self.sailable(twa, tws_kn)
        if not sailing.any():
            return drift_u, drift_v, sailing
        speed = self.polar.speed(twa, tws_kn) * (self.speed_scale * geo.KNOT_MS) * speed_factor
        sail_u, sail_v = geo.heading_to_uv(speed, heading)
        return np.where(sailing, sail_u, drift_u), np.where(sailing, sail_v, drift_v), sailing

    def with_overrides(self, **kw) -> "Vessel":
        """A validated copy with fields replaced, e.g. ``speed_scale=1.2`` for sensitivity runs.

        A mapping for ``leeway`` replaces only the coefficients it names. The polar
        is not rebuilt, so lowering ``min_twa_deg`` into its zero region is rejected.
        """
        if isinstance(kw.get("leeway"), Mapping):
            kw["leeway"] = dataclasses.replace(self.leeway, **kw["leeway"])
        return dataclasses.replace(self, **kw)


# --- configuration ------------------------------------------------------------

_SCALAR_KEYS = ("min_twa_deg", "min_sail_tws_kn", "heave_to_tws_kn", "speed_scale")
_VESSEL_KEYS = {"kind", "polar", "leeway", "sources", *_SCALAR_KEYS}
_POLAR_KEYS = {
    "parametric": {"min_twa_deg", "max_speed_kn", "wind_ratio", "angle_factors"},
    "table": {"twa_deg", "tws_kn", "speed_kn"},
}
_LEEWAY_KEYS = {f.name for f in dataclasses.fields(Leeway)}


def _check_keys(cfg, allowed, required, what):
    if not isinstance(cfg, Mapping):
        raise ValueError(f"{what} must be a mapping")
    if unknown := set(cfg) - set(allowed):
        raise ValueError(f"unknown {what} keys {sorted(unknown)}; allowed: {sorted(allowed)}")
    if missing := set(required) - set(cfg):
        raise ValueError(f"missing {what} keys {sorted(missing)}")


def _check_sources(params, sources):
    """Every parameter needs a citation under its dotted key or a parent key; no stray keys."""
    covered = {p.split(".")[0] for p in params} | set(params)
    if missing := [p for p in params if p not in sources and p.split(".")[0] not in sources]:
        raise ValueError(f"no sources entry for {missing}")
    if stray := sorted(set(sources) - covered):
        raise ValueError(f"sources name parameters the vessel does not have: {stray}")


def _polar_from_config(cfg, vessel_min_twa):
    if not isinstance(cfg, Mapping) or cfg.get("type") not in _POLAR_KEYS:
        raise ValueError(f"polar needs a type, one of {sorted(_POLAR_KEYS)}")
    keys = _POLAR_KEYS[cfg["type"]]
    _check_keys(cfg, {"type"} | keys, keys - {"min_twa_deg"}, "polar")
    if cfg["type"] == "parametric":
        return parametric_polar(float(cfg.get("min_twa_deg", vessel_min_twa)), float(cfg["max_speed_kn"]),
                                float(cfg["wind_ratio"]), cfg["angle_factors"])
    return Polar(cfg["twa_deg"], cfg["tws_kn"], cfg["speed_kn"])


def _vessel_from_config(name, cfg):
    _check_keys(cfg, _VESSEL_KEYS, {"kind", "leeway"}, "vessel")
    _check_keys(cfg["leeway"], _LEEWAY_KEYS, {"dw_slope"}, "leeway")
    scalars = {k: float(cfg[k]) for k in _SCALAR_KEYS if k in cfg}
    params = [*scalars, *(f"leeway.{k}" for k in cfg["leeway"])]
    polar = None
    if cfg.get("polar") is not None:
        polar = _polar_from_config(cfg["polar"], scalars.get("min_twa_deg", 180.0))
        params += [f"polar.{k}" for k in cfg["polar"] if k != "type"]
    sources = cfg.get("sources") or {}
    if not isinstance(sources, Mapping) or not all(isinstance(v, str) and v.strip() for v in sources.values()):
        raise ValueError("sources must map parameter names to non-empty citation strings")
    sources = {str(k): v for k, v in sources.items()}
    _check_sources(params, sources)
    leeway = Leeway(**{k: float(v) for k, v in cfg["leeway"].items()})
    return Vessel(name=name, kind=cfg["kind"], leeway=leeway, polar=polar, sources=sources, **scalars)


def load_vessels(path) -> dict[str, Vessel]:
    """Read vessel definitions (schema documented at the top of ``configs/vessels.yaml``)."""
    with open(path, encoding="utf-8") as fh:
        doc = yaml.safe_load(fh)
    if not isinstance(doc, Mapping) or not isinstance(doc.get("vessels"), Mapping):
        raise ValueError(f"{path}: expected a top-level 'vessels' mapping")
    vessels = {}
    for name, cfg in doc["vessels"].items():
        try:
            vessels[str(name)] = _vessel_from_config(str(name), cfg)
        except (TypeError, ValueError) as err:
            raise ValueError(f"{path}: vessel {name!r}: {err}") from err
    return vessels
