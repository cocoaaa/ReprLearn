import math
from geopy.geocoders import Nominatim
from typing import Tuple


def deg2rad(x):
	"""Convert the unit of x from degree to radian"""
	return x * math.pi/180.0

def getTileFromGeo(lat_deg, lng_deg, zoom):
	'''
	get tile index from geo location
	:type : float, float, int
	:rtype: tuple(int, int, int)
	'''
	x = math.floor((lng_deg + 180) / 360.0 * (2.0 ** zoom))

	tan_y = math.tan(lat_deg * (math.pi / 180.0))
	cos_y = math.cos(lat_deg * (math.pi / 180.0))
	y = math.floor( (1 - math.log(tan_y + 1.0 / cos_y) / math.pi) / 2.0 * (2.0 ** zoom) )

	return int(x), int(y), int(zoom)


def getGeoFromTile(x:int, y:int, zoom:int):
	lon_deg = x / (2.0 ** zoom) * 360.0 - 180.0
	lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * y / (2.0 ** zoom))))
	lat_deg = lat_rad * (180.0 / math.pi)
	return lat_deg, lon_deg


def getTileExtent(x, y, zoom):
	"""Computes the horizontal and vertical distance covered by the tile of size 256x256 (in meters)

    Assumes tile size 256x256
    Returns
    -------
    - (extent_y, exent_x): extent of the region coverage in latitudal and longitudal distance (meter)
    Ref: "Distance per pixel math" in https://wiki.openstreetmap.org/wiki/Zoom_levels

    """
	lat_deg, lon_deg = getGeoFromTile(x,y,zoom)
	lat_rad, lon_rad = deg2rad(lat_deg), deg2rad(lon_deg)

	C_meters = 2*math.pi*6378137 # equatorial circumference of the Earth in meters
	size_y = C_meters * math.cos(lat_rad)/2**zoom
	size_x = C_meters * math.cos(lon_rad)/2**zoom
	return size_y, size_x


def getCountryFromTile(x:int ,y: int, zoom:int) -> str:
	"""Given x,y,z tile coords, return cityname"""
	lat_deg, lng_deg = getGeoFromTile(x,y,zoom)
	geolocator = Nominatim(user_agent="temp")
	location = geolocator.reverse(f"{lat_deg}, {lng_deg}")
	city = location.address.split(" ")[-1]
	return city

def test_getCountryFromTile():
    print("shanghai: ", getCountryFromTile(13703, 6671, 14))
    print("paris: ", getCountryFromTile(8301, 5639, 14))


def parse_tileXY(
	xy_str: str,
	delimiter:str = '-',
	z: int = 14) -> Tuple[int]:
	#Parse a string of f'{lat_deg}-{lng_deg}' and return a tuple of (lat_deg, lng_deg,z_)

    x, y = list(map(int, xy_str.split(delimiter)))

    return (x, y, z)


def getCountryFromTileXY(
		xy_str: str,
	    delimiter='-',
		z: int = 14
) -> str:
    tile_xyz = parse_tileXY(xy_str, delimiter, z)

    return getCountryFromTile(*tile_xyz)

