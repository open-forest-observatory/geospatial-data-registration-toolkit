import typing
from pathlib import Path
import pyproj

PATH_TYPE = typing.Union[Path, str]

DATA_FOLDER = Path(Path(__file__).parent, "..", "data").resolve()
LAT_LON_CRS = pyproj.CRS.from_epsg(4326)
