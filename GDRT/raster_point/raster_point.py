import numpy as np
import geopandas as gpd
import rasterio as rio
from itertools import product
import shapely
from skimage.segmentation import watershed
from skimage.filters import gaussian
import matplotlib.pyplot as plt


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


def WMAE(sampled_heights, provided_heights):
    diff = np.abs(sampled_heights - provided_heights)
    weighting = np.power(provided_heights, 2)

    metric = np.sum(diff * weighting) / np.sum(weighting)

    return -metric


def find_best_shift(
    raster_file,
    points_file,
    comparison_func=corr_func,
    height_col="height",
    x_range=(-100, 100, 10),
    y_range=(-100, 100, 10),
    vis=False,
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

        ratio = np.nan
    else:
        best_idx = np.nanargmax(correlations)
        best_shift = shifts[best_idx]
        best_corr = correlations[best_idx]

        # Compute the quality metric
        img_copy = correlations_img.copy().astype(float)
        # Smooth the image prior to watershed
        img_copy = gaussian(img_copy, sigma=1)

        # Find the max of the smooth image
        first_max = np.nanmax(img_copy)

        if vis:
            plt.imshow(img_copy)
            plt.xticks(ticks=np.arange(len(x_vals)), labels=x_vals)
            plt.yticks(ticks=np.arange(len(y_vals)), labels=y_vals)
            plt.colorbar()
            plt.title("Correlation surface (unmasked)")
            plt.show()

        # Perform watershed segmentation to determine the different basins
        seg = watershed(-img_copy, connectivity=2)

        # Find the label of the optimal basin by first computing the indices and then querying the
        # value
        max_location_i = np.where(y_vals == best_shift[1])[0][0]
        max_location_j = np.where(x_vals == best_shift[0])[0][0]
        label_of_optimal = seg[max_location_i, max_location_j]

        # Mask out the parts of the image corresponding to the segmentation for the maximal value
        img_copy[seg == label_of_optimal] = np.nan

        # Compute the max after masking out the maximal basin
        secondary_max = np.nanmax(img_copy)

        # Compute the ratio of the two
        ratio = secondary_max / first_max

        if vis:
            plt.imshow(img_copy)
            plt.xticks(ticks=np.arange(len(x_vals)), labels=x_vals)
            plt.yticks(ticks=np.arange(len(y_vals)), labels=y_vals)
            plt.colorbar()
            plt.title("Correlation surface (masked)")
            plt.show()

    return {
        "best_shift": best_shift,
        "best_correlation": float(best_corr) if not np.isnan(best_corr) else np.nan,
        "correlations_img": correlations_img,
        "x_vals": x_vals,
        "y_vals": y_vals,
        "ratio": ratio,
    }
