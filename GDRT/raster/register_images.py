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

from GDRT.constants import PATH_TYPE
from GDRT.geospatial_utils import get_projected_CRS
from GDRT.raster.registration_algorithms import cv2_feature_matcher
from GDRT.raster.utils import load_geospatial_crop


def align_two_rasters(
    fixed_filename: PATH_TYPE,
    moving_filename: PATH_TYPE,
    output_filename: PATH_TYPE = None,
    region_of_interest: gpd.GeoDataFrame = None,
    target_GSD: typing.Union[None, float] = None,
    aligner_alg=cv2_feature_matcher,  # TODO consider removing the default to ensure it's set inteligently
    aligner_kwargs: dict = {},
    grayscale: bool = True,
    vis_chips: bool = True,
    vis_kwargs=dict(cmap="gray", vmin=0, vmax=255),
):
    """_summary_

    Args:
        fixed_filename (PATH_TYPE): _description_
        moving_filename (PATH_TYPE): _description_
        region_of_interest (gpd.GeoDataFrame, optional): _description_. Defaults to None.
        target_GSD (typing.Union[None, float], optional): _description_. Defaults to None.
        grayscale (bool, optional): _description_. Defaults to True.
        vis_chips (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """
    # Use the fixed dataset to determine what CRS to use
    with rio.open(fixed_filename) as fixed_dataset:
        # Reproject both datasets into the same projected CRS
        if fixed_dataset.crs.is_projected:
            working_CRS = fixed_dataset.crs
        else:
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
    )

    moving_chip, moving_transform_dict = load_geospatial_crop(
        moving_filename,
        region_of_interest=region_of_interest,
        target_CRS=working_CRS,
        target_GSD=target_GSD,
    )
    fixed_chip = np.squeeze(fixed_chip)
    moving_chip = np.squeeze(moving_chip)
    if grayscale:
        if len(fixed_chip.shape) == 3 and fixed_chip.shape[2] != 1:
            fixed_chip = cv2.cvtColor(fixed_chip, cv2.COLOR_BGR2GRAY)
        if len(moving_chip.shape) == 3 and moving_chip.shape[2] != 1:
            moving_chip = cv2.cvtColor(moving_chip, cv2.COLOR_BGR2GRAY)

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

    if vis_chips:
        warped_moving = cv2.warpAffine(
            moving_chip,
            fx2mv_window_pixel_transform[:2],
            (fixed_chip.shape[1], fixed_chip.shape[0]),
            flags=cv2.WARP_INVERSE_MAP,
            borderValue=float(np.min(moving_chip)),
        )
        f, ax = plt.subplots(1, 2)
        ax[0].imshow(fixed_chip, **vis_kwargs)
        ax[1].imshow(warped_moving, **vis_kwargs)
        ax[0].set_title("Fixed chip")
        ax[1].set_title("Moving chip\n aligned to fixed")
        plt.show()

    # At the end of the day, we want a transform from pixel coords in the moving image to geospatial ones in the fixed image

    # Go from pixel coords in the moving dataset to pixel coords in the moving window
    composite_transform = moving_transform_dict["dataset_pixels_to_window_pixels"]
    # Go from pixel coords in the moving window to pixel coords in the fixed window
    composite_transform = mv2fx_window_pixel_transform @ composite_transform
    # Go from the fixed window pixels to the to the fixed dataset pixels
    composite_transform = (
        fixed_transform_dict["window_pixels_to_geo"] @ composite_transform
    )

    logging.info("About to write transformed file")

    if output_filename is not None:
        output_filename = Path(output_filename)
        output_filename.parent.mkdir(exist_ok=True, parents=True)
        if not os.path.isfile(output_filename):
            shutil.copy(moving_filename, output_filename)

        # TODO the CRS should be examined
        with rio.open(output_filename, "r+") as dataset:
            dataset.transform = rio.guard_transform(composite_transform[:2].flatten())
    return composite_transform
