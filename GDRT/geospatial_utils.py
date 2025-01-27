import geopandas as gpd
import numpy as np
import pyproj
import rasterio as rio
import shapely
from contourpy import contour_generator


def get_projected_CRS(lat, lon, assume_western_hem=True):
    if assume_western_hem and lon > 0:
        lon = -lon
    # https://gis.stackexchange.com/questions/190198/how-to-get-appropriate-crs-for-a-position-specified-in-lat-lon-coordinates
    epgs_code = 32700 - round((45 + lat) / 90) * 100 + round((183 + lon) / 6)
    crs = pyproj.CRS.from_epsg(epgs_code)
    return crs


def mask_to_shapely(mask: np.ndarray) -> shapely.MultiPolygon:
    """Convert a binary mask to a shapely polygon representing the positive regions

    Args:
        mask (np.ndarary):
            A (n, m) array where positive values are > 0.5 and negative values are < 0.5

    Returns:
        shapely.MultiPolygon:
            A multipolygon representing the positive regions. Holes and multiple disconnected
            components are properly handled.
    """
    # This generally follows the example here:
    # https://contourpy.readthedocs.io/en/v1.3.0/user_guide/external/shapely.html#filled-contours-to-shapely

    # If mask is empty, return an empty Polygon
    if not np.any(mask):
        return shapely.Polygon()

    # Extract the contours and create a filled contour for the regions above 0.5
    filled = contour_generator(z=mask, fill_type="ChunkCombinedOffsetOffset").filled(
        0.5, np.inf
    )

    # Create a polygon for each of the disconnected regions, called chunks in ContourPy
    # This iterates over the elements in three lists, the points, offsets, and outer offsets
    chunk_polygons = [
        shapely.from_ragged_array(
            shapely.GeometryType.POLYGON, points, (offsets, outer_offsets)
        )
        for points, offsets, outer_offsets in zip(*filled)
    ]
    # Union these regions to get a single multipolygon
    multipolygon = shapely.unary_union(chunk_polygons)

    return multipolygon


def extract_bounding_polygon(raster_filename):
    # Open the dataset. This is a cheap operation.
    dataset = rio.open(raster_filename)
    # Determine which pixels have a valid mask
    mask = np.isfinite(dataset.read()[0])
    # Convert this to a polygon boundary
    polygon = mask_to_shapely(mask)

    # Apply the transformation to get into geospatial coordinates
    transformed_polygon = shapely.transform(
        polygon,
        lambda x: np.array(rio.transform.xy(dataset.transform, x[:, 1], x[:, 0])).T,
    )
    # Create a geodataframe with one geometry
    gdf = gpd.GeoDataFrame(geometry=[transformed_polygon], crs=dataset.crs)

    return gdf
