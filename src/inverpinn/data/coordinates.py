"""WGS84 lon/lat degrees <-> Almaty UTM 43N metres, always longitude first."""

import numpy as np
from pyproj import CRS, Geod, Transformer

GEOGRAPHIC_CRS = CRS.from_epsg(4326)
PROJECTED_CRS = CRS.from_epsg(32643)
_FORWARD = Transformer.from_crs(GEOGRAPHIC_CRS, PROJECTED_CRS, always_xy=True)
_INVERSE = Transformer.from_crs(PROJECTED_CRS, GEOGRAPHIC_CRS, always_xy=True)
_GEOD = Geod(ellps="WGS84")


def to_projected(longitude, latitude):
    """Return (easting,northing) in EPSG:32643 metres without normalization.

    UTM zone 43N spans 72–78°E and 0–84°N, including Almaty. Its central
    meridian is 75°E, scale factor 0.9996 and false easting 500000 m.
    Reject invalid/out-of-zone inputs rather than silently choosing another
    projection. Arrays broadcast; altitude is not transformed.
    """
    lon, lat = np.broadcast_arrays(np.asarray(longitude, float), np.asarray(latitude, float))
    if not (np.isfinite(lon).all() and np.isfinite(lat).all()
            and ((lon >= 72) & (lon <= 78) & (lat >= 0) & (lat <= 84)).all()):
        raise ValueError("Coordinates must be finite and inside UTM zone 43N (lon 72..78, lat 0..84).")
    return _FORWARD.transform(lon, lat, errcheck=True)


def to_wgs84(x, y):
    """Return (longitude,latitude) degrees for map plotting, inverse of above."""
    x, y = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float))
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Projected coordinates must be finite.")
    lon, lat = _INVERSE.transform(x, y, errcheck=True)
    to_projected(lon, lat)  # Enforce the same supported area on inverse use.
    return lon, lat


def project_wind(longitude, latitude, eastward, northward):
    r"""Return coordinate velocities (dx/dt,dy/dt) in projected metres/second.

    East/north wind is not identical to UTM grid-x/grid-y wind. Apply the local
    Jacobian J of the projection: [u_x,v_y]^T = J [u_east,v_north]^T.
    Estimate J by symmetric one-metre WGS84 geodesic displacements. This
    includes grid convergence AND local map scale; it is not just relabeling.
    Input winds are m/s. Missing winds propagate as NaN, never filled.
    """
    lon, lat, u, v = np.broadcast_arrays(*[np.asarray(a, float) for a in
        (longitude, latitude, eastward, northward)])
    to_projected(lon, lat)
    columns = []
    for positive, negative in ((90., 270.), (0., 180.)):
        lp, bp, _ = _GEOD.fwd(lon, lat, np.full(lon.shape, positive), np.ones(lon.shape))
        lm, bm, _ = _GEOD.fwd(lon, lat, np.full(lon.shape, negative), np.ones(lon.shape))
        xp, yp = _FORWARD.transform(lp, bp, errcheck=True)
        xm, ym = _FORWARD.transform(lm, bm, errcheck=True)
        columns.append(((np.asarray(xp)-xm)/2, (np.asarray(yp)-ym)/2))
    return columns[0][0]*u + columns[1][0]*v, columns[0][1]*u + columns[1][1]*v
