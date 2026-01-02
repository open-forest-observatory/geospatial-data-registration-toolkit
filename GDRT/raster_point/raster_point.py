import numpy as np
import geopandas as gpd
import rasterio as rio
from itertools import product
import shapely


def corr_func(sampled_heights, provided_heights):
    # Handle masked arrays
    if np.ma.isMaskedArray(sampled_heights):
        sampled_heights = sampled_heights.filled(np.nan)

    # Keep only points where both are finite
    mask = np.isfinite(sampled_heights) & np.isfinite(provided_heights)

    if mask.sum() < 2:
        corr = np.nan
    else:
        corr = np.corrcoef(sampled_heights[mask], provided_heights[mask])[0, 1]

    return corr


def corr_nan_weighted_func(sampled_heights, provided_heights):
    """
    This penalizes nan elements by treating them as terms with zero correlation. TODO, to penalize
    maximally, this could be changed to treating them as -1 correlation.
    """
    # Handle masked arrays
    if np.ma.isMaskedArray(sampled_heights):
        sampled_heights = sampled_heights.filled(np.nan)

    # Keep only points where both are finite
    mask = np.isfinite(sampled_heights) & np.isfinite(provided_heights)

    if mask.sum() < 2:
        corr = np.nan
    else:
        corr = np.corrcoef(sampled_heights[mask], provided_heights[mask])[0, 1]
        # Correction factor so nans count as zero correlation, bringing the overall correlation
        # closer to zero
        corr = corr * mask.sum() / len(mask)

    return corr


def find_best_shift(
    raster_file,
    points_file,
    comparison_func=corr_func,
    height_col="height",
    x_range=(-100, 100, 10),
    y_range=(-100, 100, 10),
):
    """Search for the shift (dx, dy) that maximizes the Pearson correlation between
    sampled raster elevations at shifted point locations and the provided point heights.

    Args:
        raster_file (str): Path to the raster file.
        points_file (str): Path to the points file (GeoPackage, shapefile, etc.).
        comparison_func (functional):
            The function which takes in the true and sampled heights and returns a score which is
            interpreted as higher is better.
        height_col (str): Name of the height column in the points file.
        x_range (tuple or array): If tuple (start, stop, step) it will be passed to np.arange;
                                 otherwise pass an array of x offsets to try.
        y_range (tuple or array): Same as x_range but for y offsets.

    Returns:
        dict: {
            'best_shift': (dx, dy),
            'best_correlation': float,
            'correlations_img': 2D numpy array shaped (len(x_vals), len(y_vals)),
            'x_vals': numpy array of x offsets,
            'y_vals': numpy array of y offsets
        }
    """

    # Normalize ranges to arrays
    def _to_vals(r):
        r = np.asarray(r)
        if r.ndim == 1 and r.size == 3:
            return np.arange(r[0], r[1], r[2])
        return r

    x_vals = _to_vals(x_range)
    y_vals = _to_vals(y_range)

    shifts = list(product(x_vals, y_vals))

    correlations = []

    with rio.open(raster_file) as raster:
        # Read the points, ensuring they are in the same CRS as the raster
        sample_points = gpd.read_file(points_file).to_crs(raster.crs)

        # Extract the xy locations of the points
        xy_points = shapely.get_coordinates(sample_points.geometry)
        provided_heights = sample_points[height_col].values

        for dx, dy in shifts:
            shifted_xy_points = xy_points + np.array([dx, dy])

            # sample_gen returns an iterator of values (one per point)
            sampled = np.array(
                list(rio.sample.sample_gen(raster, shifted_xy_points))
            ).squeeze()

            metric = comparison_func(provided_heights, sampled)

            correlations.append(metric)

    correlations = np.array(correlations)
    correlations_img = correlations.reshape(len(x_vals), len(y_vals)).T

    # Find best (highest) correlation, ignoring NaNs
    if np.all(np.isnan(correlations)):
        best_shift = (np.nan, np.nan)
        best_corr = np.nan
    else:
        best_idx = np.nanargmax(correlations)
        best_shift = shifts[best_idx]
        best_corr = correlations[best_idx]

    return {
        "best_shift": best_shift,
        "best_correlation": float(best_corr) if not np.isnan(best_corr) else np.nan,
        "correlations_img": correlations_img,
        "x_vals": x_vals,
        "y_vals": y_vals,
    }
