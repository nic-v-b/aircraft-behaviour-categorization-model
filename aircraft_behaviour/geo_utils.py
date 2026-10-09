"""Geospatial helpers shared by the research scripts."""

import pyproj


_GEODESIC = pyproj.Geod(ellps="WGS84")


def get_bearing_from_2pts(lon1, lat1, lon2, lat2):
    """Return the bearing using the conversion used in the published research code."""
    forward_azimuth, _back_azimuth, _distance = _GEODESIC.inv(
        lon1, lat1, lon2, lat2
    )
    if forward_azimuth < 0:
        forward_azimuth = abs(forward_azimuth) + 180
    return forward_azimuth
