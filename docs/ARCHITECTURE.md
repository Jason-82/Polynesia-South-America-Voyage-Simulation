# Architecture

This is the contract between modules. SPEC.md says *what* the simulator must
answer; this file says *how the code is shaped*. Change a signature here first,
then in code.

## Conventions (all modules)

Defined once in `src/voyage_sim/geo.py`; never re-derive them.

| Quantity | Convention |
| --- | --- |
| Longitude | degrees east, wrapped to [-180, 180). The study domain (165°W–65°W) never crosses the antimeridian. |
| Latitude | degrees north |
| Heading, bearing | compass degrees clockwise from true north, [0, 360) |
| Velocity | `(u, v)` m/s, u eastward, v northward |
| Wind direction | a wind vector points where the air goes; `geo.wind_from_deg(u, v)` is where it comes from |
| True wind angle (TWA) | `geo.true_wind_angle(heading, wind_from)`: 0 = head to wind, 180 = dead downwind |
| Time | float64 **epoch hours** (hours since 1970-01-01T00Z); `geo.to_epoch_hours` / `geo.from_epoch_hours` |
| Distances for sighting | km, great-circle |
| Speeds in config files | knots; converted to m/s on load with `geo.KNOT_MS` |
| Randomness | `voyage_sim.noise` only: draws keyed by `(seed, voyage_id, step, channel)`. No `np.random` global state, no per-batch generators. |

All per-voyage quantities are 1-D NumPy arrays of length `n` (one entry per
voyage). Functions are vectorized over voyages; no Python loop over voyages
anywhere in the hot path.

## Data flow

```
config YAML ──► run.py ──► fields.Environment ─┐
                     ├──► vessels.Vessel ──────┤
                     ├──► strategies.Strategy ─┼──► simulate.simulate() ──► VoyageResults ──► analyze
                     ├──► landfall.LandfallDetector (one SightingRaster per sighting rule)
                     └──► simulate.Launches
```

## Modules

### `geo.py` (done)
Angles, distances, motion (`advance`), unit-sphere helpers for KD-trees, time.

### `noise.py` (done)
`uniform(seed, ids, step, channel)` and `normal(...)`. Channel numbers used by
the simulator are constants in `simulate.py`.

### `fields.py`: wind and current in space and time

```python
class VectorField(Protocol):
    def sample(self, lon, lat, t) -> tuple[np.ndarray, np.ndarray]: ...
    # lon, lat, t: 1-D arrays of equal length (t in epoch hours).
    # Returns (u, v) m/s, float64. NaN where the point is outside the field's
    # spatial or temporal coverage (the simulator treats that as "left domain").

@dataclass
class UniformField:            # constant everywhere and always
    u: float
    v: float

@dataclass
class FunctionField:           # analytic fields for tests: fn(lon, lat, t) -> (u, v)
    fn: Callable

class GriddedField:
    def __init__(self, lon, lat, time, u, v, *, land_value=0.0, name=""):
        # lon (nx,), lat (ny,), time (nt,) epoch hours; u, v shape (nt, ny, nx).
        # Accepts lon in 0..360 and descending lat; normalizes to ascending lat
        # and [-180, 180) lon. nt == 1 means time-invariant.
        # Rectilinear (not necessarily uniform) grids; lookup via searchsorted.
    @classmethod
    def from_dataset(cls, ds, u, v, **kw) -> "GriddedField":
        # xarray Dataset; auto-detects coordinate names
        # (longitude|lon|x, latitude|lat|y, time|valid_time); squeezes a
        # length-1 depth dimension (GLORYS surface level).
    bounds -> (lon_min, lon_max, lat_min, lat_max, t_min, t_max)
    # sample(): bilinear in space, linear in time. NaN corners (land in ocean
    # products) are dropped and the remaining weights renormalized; if every
    # corner is NaN the value is land_value (0 m/s current on land).

class MonthlyClimatologyField:
    def __init__(self, lon, lat, u12, v12, *, land_value=0.0, name=""):
        # u12, v12 shape (12, ny, nx), index 0 = January.
        # Time interpolation: linear between mid-month centres, cyclic
        # (December <-> January), using the calendar date of t. Valid for any t.

def monthly_climatology(time, u, v) -> tuple[np.ndarray, np.ndarray]:
    # NaN-aware mean by calendar month of gridded data -> (u12, v12).

@dataclass
class Environment:
    wind: VectorField          # 10 m wind
    current: VectorField       # surface current
    current_is_climatology: bool = False   # flagged in outputs (pre-1993 runs)
```

### `vessels.py`: what a hull and rig can do

```python
@dataclass(frozen=True)
class Polar:
    twa_deg: np.ndarray   # ascending, within [0, 180]
    tws_kn: np.ndarray    # ascending, starts at 0
    speed_kn: np.ndarray  # (len(twa_deg), len(tws_kn)) boat speed through water
    def speed(self, twa_deg, tws_kn) -> np.ndarray:  # knots; bilinear, inputs clipped to table range

def parametric_polar(min_twa_deg, max_speed_kn, wind_ratio, angle_factors) -> Polar:
    # speed = min(max_speed_kn, wind_ratio * tws) * f(twa); f piecewise-linear
    # through angle_factors [(twa, factor), ...], zero below min_twa_deg.

@dataclass(frozen=True)
class Leeway:                  # Allen & Plourde (1999) form: L = slope * W10 + offset
    dw_slope: float            # downwind component per unit 10 m wind speed
    dw_offset_ms: float = 0.0
    cw_slope: float = 0.0      # crosswind magnitude per unit wind speed
    cw_offset_ms: float = 0.0
    def velocity(self, wind_u, wind_v, cw_sign) -> tuple[np.ndarray, np.ndarray]:
        # Downwind part along the wind vector; crosswind part perpendicular,
        # cw_sign = +1 to the right of downwind, -1 to the left (fixed per
        # voyage). Components floored at 0 before the sign; zero in calm air.

@dataclass(frozen=True)
class Vessel:
    name: str
    kind: str                  # "sailing" | "drift"
    leeway: Leeway             # used whenever the vessel is not sailing
    polar: Polar | None = None
    min_twa_deg: float = 180.0 # closest angle to the wind it can sail
    min_sail_tws_kn: float = 0.0   # below this: becalmed, drifts
    heave_to_tws_kn: float = inf   # at or above this: stops sailing, drifts
    speed_scale: float = 1.0   # sensitivity knob (+/-20% in SPEC)
    sources: Mapping[str, str] = {}   # parameter -> citation, for the parameter table
    can_sail -> bool
    def sailable(self, twa_deg, tws_kn) -> np.ndarray[bool]
    def water_velocity(self, heading_deg, wind_u, wind_v, cw_sign, speed_factor=1.0):
        # -> (u, v, sailing) velocity through the water.
        # wind_* is wind relative to the water (wind minus current).
        # heading NaN, or not sailable -> leeway drift. sailing is the bool mask.
    def with_overrides(self, **kw) -> "Vessel"

def load_vessels(path) -> dict[str, Vessel]    # configs/vessels.yaml
```

### `strategies.py`: choosing a heading

```python
@dataclass
class StepContext:
    lon, lat, t: np.ndarray
    wind_u, wind_v: np.ndarray        # relative to the water, m/s
    target_lon, target_lat: np.ndarray  # NaN when the voyage has no target
    vessel: Vessel
    dt_hours: float
    active: np.ndarray                # bool; only these voyages are being decided

class Strategy:
    name: str
    max_wait_days: float = 0.0        # >0 only for wait-for-wind
    def init_state(self, n) -> dict[str, np.ndarray]
    def ready_to_depart(self, state, ctx) -> np.ndarray[bool]   # default: all True
    def heading(self, state, ctx) -> np.ndarray[float]
        # Compass heading to steer, NaN = do not steer (drift / hove-to).
        # May update `state` arrays in place, only where ctx.active.

def steer_toward(desired_deg, wind_from_deg, min_twa_deg, tack, hysteresis_deg):
    # -> (heading, tack). If the desired course is at least min_twa off the wind,
    # steer it. Otherwise sail close-hauled: tack +1 is wind_from + min_twa,
    # tack -1 is wind_from - min_twa. Keep the current tack unless the other is
    # closer to the desired course by more than hysteresis_deg (tack 0 = unset).

# Implementations (names are the config keys):
#   drift            Drift()
#   downwind         Downwind(min_twa_deg=90)   never closer than max(vessel, this)
#   hold_course      HoldCourse(tack_hysteresis_deg=15)
#   wait_for_wind    WaitForWind(max_wait_days=60, margin_deg=10, min_tws_kn=5,
#                                max_tws_kn=25, ...)  then behaves as hold_course
#   latitude         LatitudeSailing(via_lat=None, lat_tol_deg=0.5,
#                                    lookahead_deg=5, final_approach_deg=2)
#                    phase 0: make the goal latitude (target's, or via_lat);
#                    phase 1: run east/west along it; phase 2: hold course to target.
#                    via_lat=-38 gives the westerlies route.
def make_strategy(name, **params) -> Strategy
```

### `landfall.py`: is land in sight?

```python
@dataclass
class Landmass:
    name: str
    group: str                 # archipelago or coast segment, e.g. "Marquesas", "Coast: Ecuador"
    height_m: float            # effective sighting height (atolls: vegetation height)
    rings: list[np.ndarray]    # each (k, 2) [lon, lat]; a (1, 2) point is allowed
    kind: str = "island"       # "island" | "atoll" | "coast"

@dataclass(frozen=True)
class SightingRule:
    name: str
    mode: str = "height"       # "height" | "uniform"
    radius_km: float = 40.0    # uniform mode
    eye_height_m: float = 3.0
    refraction_k: float = 0.13
    scale: float = 1.0         # multiplies height-based radii (sensitivity)
    floor_km: float = 0.0
    cap_km: float = inf
    def radius_km_for(self, height_m) -> np.ndarray

def geometric_range_km(target_height_m, eye_height_m=3.0, refraction_k=0.13)
    # sqrt(2 R' h_target) + sqrt(2 R' h_eye), R' = R / (1 - k)

class SightingRaster:
    # margin_km[j, i] = min over landmasses of (distance to its coastline - its radius),
    # capped at cap_km; index[j, i] = landmass of that minimum (-1 if none within cap).
    @classmethod
    def build(cls, landmasses, rule, bounds, res_deg=0.05, densify_km=1.0, cap_km=250.0)
        # bounds = (lon_min, lon_max, lat_min, lat_max). Distances are exact
        # great-circle distances from cell centres to coastline vertices
        # densified to <= densify_km (scipy cKDTree on unit vectors), computed
        # per landmass only within (radius + cap_km) of it.
    def query(self, lon, lat) -> (margin_km, index)
        # Bilinear margin; index from the nearest cell. Outside bounds: (cap_km, -1).
    def save(self, path) / load(cls, path)    # .npz cache

class LandfallDetector:
    def __init__(self, landmasses, rasters, substep_km=2.0)
    groups: list[str]; landmass_group: np.ndarray[int]   # landmass index -> group index
    def check_segment(self, lon0, lat0, lon1, lat1, origin_group, armed, active):
        # -> (hit, frac, landmass, armed_new), each shaped (n_rules, n).
        # Swept-path test: a segment is skipped only if
        #   min(margin_start, margin_end) > 0.75 * length + raster cell size,
        # which guarantees no sighting along it; otherwise it is sampled every
        # substep_km and the first sighting is located by linear interpolation
        # of the margin between samples. frac in [0, 1] is where along the
        # segment land first came into sight.
        # Home exclusion: sightings of the voyage's origin group are ignored
        # until the voyage has once been out of sight of all land (armed).

def circle_ring(lon, lat, radius_km, n=64) -> np.ndarray      # test / placeholder geometry
def load_catalog(path) -> list[Landmass]                        # configs/landmasses.yaml
def attach_gshhg(landmasses, shp_path) -> list[Landmass]        # real shorelines (optional)
```

### `simulate.py`: the integrator

```python
@dataclass
class Launches:
    voyage_id: np.ndarray      # int64, stable; keys the random streams
    lon, lat: np.ndarray       # departure point (just offshore)
    t0: np.ndarray             # epoch hours of launch (waiting starts here)
    origin_group: np.ndarray   # int index into detector.groups
    target_lon, target_lat: np.ndarray   # NaN if no target

@dataclass
class SimConfig:
    dt_hours: float = 6.0
    max_days_at_sea: float = 90.0   # horizon; shorter supply limits are applied afterwards
    integrator: str = "midpoint"    # "midpoint" (RK2) | "euler"
    speed_sd: float = 0.10          # per-voyage speed factor ~ N(1, sd), floored at 0.3
    heading_sd_deg: float = 10.0    # per-step steering error
    diffusivity_m2s: float = 0.0    # random-walk dispersion K: step sd = sqrt(2 K dt) per axis
    track_every: int = 4            # record a position every N steps (0 = none)
    seed: int = 0

# Status codes
WAITING, AT_SEA, END_ALL_SIGHTED, END_HORIZON, DID_NOT_SAIL, LEFT_DOMAIN = range(6)

def simulate(env, vessel, strategy, launches, detector, cfg) -> VoyageResults
```

One step, per voyage (SPEC "Model design"):

1. **In port** (wait-for-wind only): read the wind at the port; depart if the
   strategy says so; give up (`DID_NOT_SAIL`) after `max_wait_days`. Waiting
   does not use supplies.
2. **At sea**: read wind and current; wind relative to water = wind − current;
   the strategy picks a heading; add steering error; velocity over ground =
   vessel's water velocity + current (+ random-walk term). Midpoint
   integration re-reads the fields at the half step with the same heading.
3. **Land in sight?** Swept-segment test for every sighting rule not yet
   satisfied; record the time (`days at sea`, interpolated within the step),
   landmass and position of the first sighting under each rule.
4. **Supplies left?** Only after the landfall check: a voyage ends at the horizon
   (`END_HORIZON`), or once every rule has sighted land (`END_ALL_SIGHTED`).

Because a voyage's track does not depend on the sighting rule or the supply
limit, one run serves every radius and every supply limit: success under rule
r with S days of supplies is `hit_days[r] <= S`. A sighting exactly at the limit
counts.

```python
@dataclass
class VoyageResults:
    voyage_id, launch_t, depart_t, end_t, end_lon, end_lat, days_at_sea, status
    hit_days, hit_landmass, hit_lon, hit_lat     # (n_rules, n); NaN / -1 when no sighting
    rule_names, landmass_names, landmass_groups, group_names
    tracks: np.ndarray | None                    # (n, n_rec, 2) float32 lon/lat
    track_days: np.ndarray | None                # (n_rec,) days since launch
    current_is_climatology: bool
    meta: dict                                   # config echo, vessel, strategy, seed, version
    def to_frame(self) -> pd.DataFrame
    def save(self, path) / load(cls, path)       # .npz + JSON meta
```

### `enso.py`: climate-state labels

```python
def parse_cpc_oni(text) -> pd.Series          # CPC oni.ascii.txt -> ONI by centre month
def parse_psl_monthly(text) -> pd.Series      # PSL year + 12 values format, missing flagged
def running_mean_3(series) -> pd.Series       # centred 3-month mean
def enso_phases(oni, threshold=0.5, min_run=5) -> pd.Series   # "el_nino" | "la_nina" | "neutral"
def label_times(t_hours, phases) -> np.ndarray  # phase of each launch's calendar month
```

### `analyze.py`: odds, durations, intervals

```python
def success(results, rule, supply_days, dest_groups=None) -> np.ndarray[bool]
def cluster_bootstrap(success, clusters, n_boot=2000, seed=0, level=0.95) -> dict
    # Resample clusters (launch years) with replacement; ratio of sums;
    # percentile interval. Returns p, lo, hi, n, n_clusters.
def duration_summary(days) -> dict            # median, p90, n
def summarize(results, labels, rules, supplies, dest_groups, by) -> pd.DataFrame
```

### `fetch.py`: data access (network; never called by tests)

ERA5 via `cdsapi`, GLORYS via `copernicusmarine.subset`, GSHHG and ENSO
indices via HTTPS. Every function has a `dry_run` that returns the requests it
would make; tests check those requests only. Imports of `cdsapi` and
`copernicusmarine` are lazy so the core runs without them.

### `run.py`: one command per scenario grid

`voyage-sim run configs/<grid>.yaml` builds everything from the config, runs
each cell, writes `outputs/<name>/` (results `.npz`, `summary.csv`, the
resolved config), and gives identical bytes on rerun.
