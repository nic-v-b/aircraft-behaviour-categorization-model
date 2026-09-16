import datetime

import pyproj
import pytest

from aircraft_behaviour.geo_utils import get_bearing_from_2pts
from aircraft_behaviour.time_utils import display_time, unix_to_utc


def test_display_time_formats_multiple_units():
    assert display_time(3661, granularity=2) == "1 hour, 1 minute"


def test_unix_to_utc_epoch():
    assert unix_to_utc(0) == datetime.datetime(1970, 1, 1)


@pytest.mark.parametrize(
    ("lon2", "lat2", "expected"),
    [
        (1.0, 0.0, 90.0),
        (-1.0, 0.0, 270.0),
        (0.0, 1.0, 0.0),
    ],
)
def test_cardinal_bearings(lon2, lat2, expected):
    bearing = get_bearing_from_2pts(0.0, 0.0, lon2, lat2)
    assert bearing == pytest.approx(expected, abs=1e-6)


def test_bearing_preserves_original_negative_azimuth_conversion():
    geodesic = pyproj.Geod(ellps="WGS84")
    original_azimuth, _back, _distance = geodesic.inv(0.0, 0.0, -1.0, 1.0)
    expected = abs(original_azimuth) + 180 if original_azimuth < 0 else original_azimuth

    assert get_bearing_from_2pts(0.0, 0.0, -1.0, 1.0) == pytest.approx(expected)
