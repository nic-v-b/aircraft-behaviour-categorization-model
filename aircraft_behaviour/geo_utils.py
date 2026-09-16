"""Geospatial helpers shared by the research scripts."""

import pyproj


_GEODESIC = pyproj.Geod(ellps="WGS84")


def get_bearing_from_2pts(lon1, lat1, lon2, lat2):
    """Return forward bearing in degrees in the range [0, 360)."""
    forward_azimuth, _back_azimuth, _distance = _GEODESIC.inv(
        lon1, lat1, lon2, lat2
    )
    return forward_azimuth % 360
