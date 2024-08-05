import logging
from pathlib import Path

import cv2
import numpy as np
import SimpleITK as sitk
from matplotlib import pyplot as plt


def get_best_shift(src_pts, dst_pts):
    shift = np.mean(src_pts - dst_pts, axis=0, keepdims=True)
    return shift


def get_inlier_mask(src_pts, dst_pts, shift, threshold):
    shifted_src = src_pts - shift

    diff = dst_pts - shifted_src
    error = np.linalg.norm(diff, axis=1)
    mask = error < threshold

    return mask


def shift_only_RANSAC(
    src_pts, dst_pts, reprojection_threshold, max_iters, subset_size=3
):
    src_pts = src_pts[:, 0, :]
    dst_pts = dst_pts[:, 0, :]

    assert src_pts.shape[0] == dst_pts.shape[0]

    n_points = src_pts.shape[0]

    best_mask = np.zeros(n_points).dtype(bool)

    for _ in range(max_iters):
        selected_inds = np.random.choice(n_points, size=subset_size)
        selected_src = src_pts[selected_inds]
        selected_dst = dst_pts[selected_inds]

        # TODO do an L2 minimizer here
        shift = get_best_shift(selected_src, selected_dst)
        inlier_mask = get_inlier_mask(
            src_pts=src_pts,
            dst_pts=dst_pts,
            shift=shift,
            threshold=reprojection_threshold,
        )

        if np.sum(inlier_mask) > np.sum(best_mask):
            print(f"Updating most inliers to be {np.sum(best_mask)}")
            best_mask = inlier_mask

    inlier_src_pts = src_pts[best_mask]
    inlier_dst_pts = dst_pts[best_mask]

    # Compute a final shift with the inliears
    final_shift = get_best_shift(src_pts=inlier_src_pts, dst_pts=inlier_dst_pts)

    final_mask = get_inlier_mask(
        src_pts=src_pts,
        dst_pts=dst_pts,
        shift=final_shift,
        threshold=reprojection_threshold,
    )

    print(f"Final number of inliears is {np.sum(final_mask)}")

    # TODO the should be a final refinement step with all the inliers

    best_transform = np.array(
        [[1, 0, final_shift[0, 0]], [0, 1, final_shift[0, 1]], [0, 0, 1]]
    )
    final_mask = np.expand_dims(final_mask, 1).astype(np.uint8)

    return best_transform, best_mask


def cv2_feature_matcher(
    img1,
    img2,
    min_match_count=10,
    vis_matches=True,
    ransac_threshold=4.0,
    ransac_max_iters=10000,
    only_estimate_shift=True,
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
        if only_estimate_shift:
            M, mask = shift_only_RANSAC(
                src_pts=src_pts,
                dst_pts=dst_pts,
                reprojection_threshold=ransac_threshold,
                max_iters=ransac_max_iters,
            )
        else:
            M, mask = cv2.estimateAffinePartial2D(
                src_pts,
                dst_pts,
                method=cv2.RANSAC,
                ransacReprojThreshold=ransac_threshold,
                maxIters=ransac_max_iters,
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
        plt.imshow(img3, "gray")
        plt.title("Inliear feature matches")
        plt.show()

    return M


def sitk_intensity_registration(
    fixed_img: np.ndarray,
    moving_img: np.ndarray,
    align_means: bool = True,
    initial_translation=[0.0, 0.0],
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
    R.SetInitialTransform(sitk.TranslationTransform(2, initial_translation))
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

    # Show the overlapped images
    if vis:
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(fixed)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(0)
        resampler.SetTransform(estimated_transform)

        out = resampler.Execute(moving)

        simg1 = sitk.Cast(sitk.RescaleIntensity(fixed), sitk.sitkUInt8)
        simg2 = sitk.Cast(sitk.RescaleIntensity(out), sitk.sitkUInt8)
        zeros_like = sitk.GetImageFromArray(
            np.zeros_like(sitk.GetArrayFromImage(simg1))
        )
        cimg = sitk.Compose(simg1, simg2, zeros_like)
        cimg = sitk.GetArrayFromImage(cimg)

        plt.imshow(cimg)
        plt.title("Aligned images composite\n Fixed (red) moving (green)")
        plt.show()

    # Compute the transform matrix to return
    # TODO if we introduce other classes of transforms we'll have to make this section more generic
    translation = estimated_transform.GetOffset()
    M = np.eye(3)
    M[0, 2] = translation[0]
    M[1, 2] = translation[1]

    return M
