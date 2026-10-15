"""Near-shore relief along the Pacific coast of South America from Copernicus GLO-90 DSM.

For each 0.1 deg latitude band, max elevation of mainland pixels within D km of the
Pacific (ocean = DSM value 0, connected to the western edge of the mosaic).
"""
import json, math, os, subprocess, sys
import numpy as np, rasterio
from scipy import ndimage

DEM, NE, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
PX = 1 / 1200.0  # 3 arcsec
DISTS = (5, 10, 20, 40)

# coast longitude range per 1-deg band from NE 10m coastline (mainland only, lon > -82)
coast = json.load(open(os.path.join(NE, "ne_10m_coastline.geojson")))["features"]
bandlon = {}
for f in coast:
    g = f["geometry"]
    lines = g["coordinates"] if g["type"] == "MultiLineString" else [g["coordinates"]]
    for L in lines:
        for lon, lat in L:
            if -26 <= lat <= 7.3 and -81.5 <= lon <= -69.5:
                b = math.floor(lat)
                lo, hi = bandlon.get(b, (999, -999))
                bandlon[b] = (min(lo, lon), max(hi, lon))


def tile(la, lo):
    n = f"Copernicus_DSM_COG_30_{'N' if la >= 0 else 'S'}{abs(la):02d}_00_{'E' if lo >= 0 else 'W'}{abs(lo):03d}_00_DEM"
    p = os.path.join(DEM, n + ".tif")
    if not os.path.exists(p) and not os.path.exists(p + ".missing"):
        r = subprocess.run(["curl", "-sS", "-m", "300", "-o", p, "-w", "%{http_code}",
                            f"https://copernicus-dem-90m.s3.amazonaws.com/{n}/{n}.tif"], capture_output=True, text=True)
        if r.stdout.strip() != "200":
            if os.path.exists(p):
                os.remove(p)
            open(p + ".missing", "w").close()
    return p if os.path.exists(p) else None


rows = []
for b in sorted(bandlon, reverse=True):
    lo_c, hi_c = bandlon[b]
    # Pacific coast of Colombia only south of the Panama border (7.21N); exclude Caribbean
    lon0 = math.floor(lo_c - 0.35) if b != 7 else -79
    lon1 = math.floor(hi_c + 0.6)
    lat0, lat1 = b - 1, b + 2  # mosaic three rows for distance context
    W = (lon1 - lon0 + 1) * 1200
    H = (lat1 - lat0) * 1200
    M = np.zeros((H, W), np.float32)
    have = np.zeros((H, W), bool)
    for la in range(lat0, lat1):
        for lo in range(lon0, lon1 + 1):
            p = tile(la, lo)
            if p is None:
                continue
            with rasterio.open(p) as r:
                a = r.read(1)
                if a.shape != (1200, 1200):
                    continue
            r0 = (lat1 - 1 - la) * 1200
            c0 = (lo - lon0) * 1200
            M[r0:r0 + 1200, c0:c0 + 1200] = a
            have[r0:r0 + 1200, c0:c0 + 1200] = True
    ocean = (M <= 0) | ~have
    lab, n = ndimage.label(ocean)
    west = set(np.unique(lab[:, 0])) - {0}
    pac = np.isin(lab, list(west))
    land = ~ocean
    llab, ln = ndimage.label(land)
    east = set(np.unique(llab[:, -1])) - {0}
    main = np.isin(llab, list(east))
    lat_c = b + 0.5
    sy = 92.6  # m per pixel N-S (3 arcsec)
    sx = 92.6 * math.cos(math.radians(lat_c))
    dist = ndimage.distance_transform_edt(~pac, sampling=(sy, sx)) / 1000.0
    # center row only
    r_top = (lat1 - (b + 1)) * 1200
    for k in range(10):
        top_lat = b + 1 - k * 0.1
        rs = slice(r_top + k * 120, r_top + (k + 1) * 120)
        mid = top_lat - 0.05
        if b == 7 and mid > 7.2:
            continue
        rec = {"lat": round(mid, 2)}
        for D in DISTS:
            m = main[rs] & (dist[rs] <= D)
            if m.any():
                z = np.where(m, M[rs], -np.inf)
                i, j = np.unravel_index(np.argmax(z), z.shape)
                rec[f"max{D}"] = float(z[i, j])
                rec[f"lon{D}"] = round(lon0 + (j + 0.5) * PX, 3)
            else:
                rec[f"max{D}"] = None
        # coast longitude: westmost mainland pixel adjacent to the Pacific in this band
        cm = main[rs] & (dist[rs] <= 0.2)
        if cm.any():
            jj = np.nonzero(cm.any(axis=0))[0]
            rec["coast_lon_w"] = round(lon0 + (jj.min() + 0.5) * PX, 3)
        rows.append(rec)
    print(b, lon0, lon1, "done", flush=True)

json.dump(rows, open(OUT, "w"), indent=0)
