import argparse
import logging
import os
import shutil
import sys
import typing
from distutils.version import StrictVersion as VS
from pathlib import Path

import cv2
import geopandas as gpd
import itk
import numpy as np
import rasterio as rio
import SimpleITK as sitk
from matplotlib import pyplot as plt

from GDRT.constants import PATH_TYPE
from GDRT.geospatial_utils import get_projected_CRS
from GDRT.raster.utils import load_geospatial_crop


def cv2_feature_matcher(
    img1, img2, min_match_count=10, vis_matches=True, ransac_threshold=4.0
):
    # https://docs.opencv.org/3.4/d1/de0/tutorial_py_feature_homography.html
    # Initiate SIFT detector
    sift = cv2.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    # store all the good matches as per Lowe's ratio test.
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    if len(good) > min_match_count:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        M, mask = cv2.estimateAffinePartial2D(
            src_pts,
            dst_pts,
            method=cv2.RANSAC,
            ransacReprojThreshold=ransac_threshold,
            maxIters=10000,
        )
        # Make a 3x3 matrix
        M = np.concatenate((M, np.array([[0, 0, 1]])))

    else:
        logging.error(
            "Not enough matches are found - {}/{}".format(len(good), min_match_count)
        )
        return None

    if vis_matches:
        matchesMask = mask.astype(np.int32).ravel().tolist()
        h, w = img1.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(
            -1, 1, 2
        )
        dst = cv2.perspectiveTransform(pts, M)
        img2 = cv2.polylines(img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

        draw_params = dict(
            matchColor=(0, 255, 0),  # draw matches in green color
            singlePointColor=None,
            matchesMask=matchesMask,  # draw only inliers
            flags=2,
        )
        img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)
        plt.imshow(img3, "gray"), plt.show()

    return M


def sitk_intensity_registration(
    fixed_img: np.ndarray,
    moving_img: np.ndarray,
    align_means: bool = True,
    vis: bool = True,
    vis_metric_values: bool = True,
):
    """Align two images by trying to match the intensity. This is done using a feature-free, gradient-based method.

    Args:
        fixed_img (np.ndarray): fixed image, assumed to be (w,h)
        moving_img (np.ndarray): moving image, assumed to be (w,h)
        unitize (bool, optional): Shift each chip to zero mean and normalized by the standard diviation across the two shifted images. Defaults to True.
        vis (bool, optional): Show the ITK-specific visualization results. Defaults to True.

    Returns:
        M (np.ndarray): The (3x3) transform mapping from a pixel in the fixed_img to a pixel in the moving_img
    """
    # Create a copy to avoid modifying the inputs
    fixed_img = fixed_img.copy()
    moving_img = moving_img.copy()

    # For numerical reasons, we want the datasets to be roughly zero-centered
    # The difference is whether we shift each image based on its own mean (aligning them both to zero)
    # or shift based on the average of the two, preserving the difference between them.
    if align_means:
        fixed_img -= np.mean(fixed_img)
        moving_img -= np.mean(moving_img)
    else:
        combined_mean = np.mean(
            np.concatenate((fixed_img.flatten(), moving_img.flatten()))
        )
        fixed_img -= combined_mean
        moving_img -= combined_mean

    # Normalize both images based on the standard deviation of both datasets combined
    combined_std = np.std(np.concatenate((fixed_img.flatten(), moving_img.flatten())))
    fixed_img /= combined_std
    moving_img /= combined_std

    # The following chunk is largely taken from the SimpleITK docs
    # https://simpleitk.readthedocs.io/en/master/link_ImageRegistrationMethod3_docs.html

    # Cast both images into the SITK type
    fixed = sitk.GetImageFromArray(fixed_img)
    moving = sitk.GetImageFromArray(moving_img)

    # Create a registration method object
    R = sitk.ImageRegistrationMethod()
    # Set up the error metric
    R.SetMetricAsMeanSquares()
    # Set up the optimizer
    R.SetOptimizerAsRegularStepGradientDescent(
        learningRate=2.0,
        minStep=1e-4,
        numberOfIterations=500,
        gradientMagnitudeTolerance=1e-8,
    )
    # TODO figure out what this does
    R.SetOptimizerScalesFromIndexShift()
    # Define what class of transforms.
    # TODO look into other options and make this more general
    R.SetInitialTransform(sitk.TranslationTransform(2))
    # Set the interpolator for the shifted images
    R.SetInterpolator(sitk.sitkLinear)
    # Add a logging callback

    metric_values = []
    transform_values = []

    if vis_metric_values:

        def logging_callback(method):
            metric_values.append(method.GetMetricValue())
            transform_values.append(method.GetOptimizerPosition())

        R.AddCommand(sitk.sitkIterationEvent, lambda: logging_callback(R))

    # Actually perform the optimization routine for registration
    estimated_transform = R.Execute(fixed, moving)

    logging.info(estimated_transform)
    logging.info(
        f"Optimizer stop condition: {R.GetOptimizerStopConditionDescription()}"
    )
    logging.info(f" Iteration: {R.GetOptimizerIteration()}")
    logging.info(f" Final metric value: {R.GetMetricValue()}")

    # Show the metric values versus iteration
    if vis_metric_values:
        plt.plot(metric_values)
        plt.xlabel("Optimization iteration")
        plt.ylabel("Metric value")
        plt.show()

    # Visualization
    if vis:
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(fixed)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(0)
        resampler.SetTransform(estimated_transform)

        out = resampler.Execute(moving)

        simg1 = sitk.Cast(sitk.RescaleIntensity(fixed), sitk.sitkUInt8)
        simg2 = sitk.Cast(sitk.RescaleIntensity(out), sitk.sitkUInt8)
        cimg = sitk.Compose(simg1, simg2, simg1 // 2.0 + simg2 // 2.0)
        cimg = sitk.GetArrayFromImage(cimg)
        f, ax = plt.subplots(1, 2)
        ax[0].imshow(sitk.GetArrayFromImage(simg1))
        ax[1].imshow(sitk.GetArrayFromImage(simg2))
        plt.show()
        plt.imshow(cimg)
        plt.show()
    # Compute the transform matrix to return
    # TODO if we introduce other classes of transforms we'll have to make this section more generic
    translation = estimated_transform.GetOffset()
    M = np.eye(3)
    M[0, 2] = translation[0]
    M[1, 2] = translation[1]

    return M


def align_two_rasters(
    fixed_filename: PATH_TYPE,
    moving_filename: PATH_TYPE,
    output_filename: PATH_TYPE = None,
    region_of_interest: gpd.GeoDataFrame = None,
    target_GSD: typing.Union[None, float] = None,
    aligner_alg=cv2_feature_matcher,
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
        )
        f, ax = plt.subplots(1, 2)
        ax[0].imshow(fixed_chip, **vis_kwargs)
        ax[1].imshow(warped_moving, **vis_kwargs)
        plt.show()

        for alpha in np.arange(0.2, 0.81, 0.2):
            plt.imshow(
                (alpha * fixed_chip + (1 - alpha) * warped_moving) / 2, **vis_kwargs
            )
            plt.show()

        vis_img = np.zeros((fixed_chip.shape[0], fixed_chip.shape[1], 3))
        vis_img[..., 0] = fixed_chip
        vis_img[..., 1] = warped_moving
        plt.imshow(vis_img.astype(np.uint8))
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
