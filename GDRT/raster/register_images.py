import logging
import os
import shutil
import typing
from pathlib import Path

import cv2
import geopandas as gpd
import numpy as np
import rasterio as rio
from matplotlib import pyplot as plt
from scientific_python_utils.geospatial import get_projected_CRS

from GDRT.constants import PATH_TYPE
from GDRT.geospatial_utils import (
    extract_bounding_polygon,
    extract_largest_oriented_rectangle,
)
from GDRT.raster.utils import load_geospatial_crop


def align_two_rasters(
    fixed_filename: PATH_TYPE,
    moving_filename: PATH_TYPE,
    aligner_alg,
    region_of_interest: gpd.GeoDataFrame = None,
    target_GSD: typing.Union[None, float] = None,
    aligner_kwargs: dict = {},
    grayscale: bool = True,
    vis_chips: bool = True,
    vis_kwargs=dict(cmap="gray", vmin=0, vmax=255),
) -> dict:
    """Determine the transform between two geospatial rasters

    Args:
        fixed_filename (PATH_TYPE):
            Path to the raster with the desired geospatial registration
        moving_filename (PATH_TYPE):
            Path to the raster to register to the fixed_filename raster
        region_of_interest (gpd.GeoDataFrame, optional):
            Region of interest to use for registration. If not provided, it will be computed from
            the overlapping valid regions in the two input rasters. Defaults to None.
        target_GSD (typing.Union[None, float], optional):
            Ground sample distance to use for registration, in meters. Lower values lead to finer
            resolution, which is generally more computationally intensive. Defaults to None.
        grayscale (bool, optional):
            Should the rasters be converted to grayscale before registration. Defaults to True.
        vis_chips (bool, optional):
            Should the registered chips be visualized. Defaults to True.

    Returns:
        dict:
            "pixel_shift":
                The 3x3 transform in pixels. This maps from one chip to the other.
            "geospatial_shift":
                The 3x3 transform in units of the fixed raster.
            "composite_transform":
                The 3x3 transform mapping from a pixel in the moving image to geospatial units, now
                registered to the fixed raster
    """
    # If no region of interest is specified, compute it as the maximum overlapping region
    if region_of_interest is None:
        fixed_gdf = extract_bounding_polygon(fixed_filename)
        moving_gdf = extract_bounding_polygon(moving_filename)
        intersection = fixed_gdf.intersection(moving_gdf)
        # Base the rasterization resolution off the registration resolution so it is a reasonable
        # ballpark for the overlapping region
        region_of_interest = extract_largest_oriented_rectangle(
            intersection, raster_resolution=target_GSD * 4
        )
        # Make the region slightly more conservative to avoid weird artificats at tile boundaries
        region_of_interest.geometry = region_of_interest.buffer(-(target_GSD * 20))

    # Use the fixed dataset to determine what CRS to use
    with rio.open(fixed_filename) as fixed_dataset:
        # If the fixed dataset is projected, use that CRS
        if fixed_dataset.crs.is_projected:
            working_CRS = fixed_dataset.crs
        else:
            # Else, determine the appropriate projected CRS for that lat, lon
            working_CRS = get_projected_CRS(
                lat=fixed_dataset.transform.c, lon=fixed_dataset.transform.f
            )

    # Extract an image chip from each input image, corresponding to the region of interest
    # TODO make sure that a None ROI loads the whole image
    fixed_chip, fixed_transform_dict = load_geospatial_crop(
        fixed_filename,
        region_of_interest=region_of_interest,
        target_CRS=working_CRS,
        target_GSD=target_GSD,
        grayscale=grayscale,
    )
    moving_chip, moving_transform_dict = load_geospatial_crop(
        moving_filename,
        region_of_interest=region_of_interest,
        target_CRS=working_CRS,
        target_GSD=target_GSD,
        grayscale=grayscale,
    )

    # Show the two crops before alignment
    if vis_chips:
        _, ax = plt.subplots(1, 2)
        # TODO make these bounds more robust
        plt.colorbar(ax[0].imshow(fixed_chip, **vis_kwargs), ax=ax[0])
        plt.colorbar(ax[1].imshow(moving_chip, **vis_kwargs), ax=ax[1])
        ax[0].set_title("Fixed")
        ax[1].set_title("Moving")
        plt.show()

    # This is the potentially expensive step where we actually estimate a transform
    fx2mv_window_pixel_transform = aligner_alg(
        fixed_chip, moving_chip, **aligner_kwargs
    )
    mv2fx_window_pixel_transform = np.linalg.inv(fx2mv_window_pixel_transform)

    # Show the two chips after alignment
    if vis_chips:
        # Warp the moving chip into the coordinates of the fixed one
        warped_moving = cv2.warpAffine(
            moving_chip,
            fx2mv_window_pixel_transform[:2],
            (fixed_chip.shape[1], fixed_chip.shape[0]),
            flags=cv2.WARP_INVERSE_MAP,
            borderValue=float(np.min(moving_chip)),
        )
        _, ax = plt.subplots(1, 2)
        ax[0].imshow(fixed_chip, **vis_kwargs)
        ax[1].imshow(warped_moving, **vis_kwargs)
        ax[0].set_title("Fixed chip")
        ax[1].set_title("Moving chip\n aligned to fixed")
        plt.show()

    # Go from the pixels in the moving dataset to the geospatial reference defined by the fixed
    # dataset. This is equivalent to the dataset transform for the moving dataset, after being
    # aligned with transform produced by the registration algorithm
    updated_moving_transform = (
        fixed_transform_dict["window_pixels_to_geo"]
        @ mv2fx_window_pixel_transform
        @ moving_transform_dict["dataset_pixels_to_window_pixels"]
    )
    # Take a geospatial point, transform it into the window coordinates, transform it based on the
    # registration, and transform back to geospatial. This would be applied as a left multiplication
    # on the transform for the moving dataset
    geospatial_mv2fx_transform = (
        moving_transform_dict["window_pixels_to_geo"]
        @ mv2fx_window_pixel_transform
        @ moving_transform_dict["geo_to_window_pixels"]
    )

    # Create a dictionary of all the important transforms
    info_dict = {
        "updated_moving_transform": updated_moving_transform,
        "geospatial_mv2fx_transform": geospatial_mv2fx_transform,
    }
    return info_dict
