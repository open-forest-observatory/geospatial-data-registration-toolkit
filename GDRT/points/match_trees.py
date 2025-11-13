import itertools
import typing
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely
from shapely.affinity import translate

from matplotlib import collections as mc

from GDRT.geospatial_utils import ensure_projected_CRS


# Taken from here:
# https://stackoverflow.com/questions/6430091/efficient-distance-calculation-between-n-points-and-a-reference-in-numpy-scipy
# This is drop-in replacement for scipy.cdist
def cdist(x, y):
    """
    Compute pair-wise distances between points in x and y.

    Parameters:
        x (ndarray): Numpy array of shape (n_samples_x, n_features).
        y (ndarray): Numpy array of shape (n_samples_y, n_features).

    Returns:
        ndarray: Numpy array of shape (n_samples_x, n_samples_y) containing
        the pair-wise distances between points in x and y.
    """
    # Reshape x and y to enable broadcasting
    x_reshaped = x[:, np.newaxis, :]  # Shape: (n_samples_x, 1, n_features)
    y_reshaped = y[np.newaxis, :, :]  # Shape: (1, n_samples_y, n_features)

    # Compute pair-wise distances using Euclidean distance formula
    pairwise_distances = np.sqrt(np.sum((x_reshaped - y_reshaped) ** 2, axis=2))

    return pairwise_distances


def find_best_shift(
    field_trees: gpd.GeoDataFrame,
    drone_trees: gpd.GeoDataFrame,
    obs_bounds: gpd.GeoDataFrame,
    objective_function: typing.Callable,
    search_window: float = 50,
    search_increment: float = 2,
    base_shift: typing.Tuple[float] = (0, 0),
    vis: bool = False,
) -> np.array:
    """
    Compute the shift for the observed trees that minimizes the mean distance between observed trees
    and the matching drone trees.


    Args:
        field_trees (gpd.GeoDataFrame):
            Dataframe of field trees
        drone_trees (gpd.GeoDataFrame):
            Dataframe of drone trees
        obs_bounds: (gpd.GeoDataFrame):
            GeoDataFrame with a single polygon geometry representing the surveyed area corresponding
            to the field survey.
        objective_function (function):
            A function that takes the drone trees and shifted field trees and computes a score.
            Higher scores imply better alignment.
        search_window (float, optional):
            Distance in meters to perform grid search. Defaults to 50.
        search_increment (float, optional):
            Increment in meters for grid search. Defaults to 2.
        base_shift_x (float, optional):
            Center the grid search around shifting the x of observations this much. Defaults to 0.
        base_shift_y (float, optional):
            Center the grid search around shifting the y of observations this much. Defaults to 0.
        vis (bool, optional):
            Visualize a scatter plot of the mean closest distance to drone trees for each shift.
            Defaults to False.

    Returns:
        np.array:
            The [x, y] shift that should be applied to the field trees to align them with the
            drone trees
    """
    # Build the shifts. Note that our eventual goal is to recover a shift for the observed trees,
    # assuming the drone trees remain fixed
    x_shifts = np.arange(
        start=base_shift[0] - search_window,
        stop=base_shift[0] + search_window,
        step=search_increment,
    )
    y_shifts = np.arange(
        start=base_shift[1] - search_window,
        stop=base_shift[1] + search_window,
        step=search_increment,
    )
    shifts = [shift for shift in (itertools.product(x_shifts, y_shifts))]

    # Iterate over the shifts and compute the objective function when applied to the shifted field
    # trees
    objective_values = []
    for shift in shifts:
        # Shift the field points and observation bounds
        shifted_field_trees = field_trees.copy()
        shifted_obs_bounds = obs_bounds.copy()

        shifted_field_trees.geometry = shifted_field_trees.translate(
            xoff=shift[0], yoff=shift[1]
        )
        shifted_obs_bounds.geometry = shifted_obs_bounds.translate(
            xoff=shift[0], yoff=shift[1]
        )

        # Compute the quality of this shift
        objective_values.append(
            objective_function(shifted_field_trees, drone_trees, shifted_obs_bounds)
        )

    if vis:
        # Extract the x and y components of the shifts
        x = [shift[0] for shift in shifts]
        y = [shift[1] for shift in shifts]

        # Create a scatter plot of the shifts versus the quailty of the alignment
        plt.scatter(x, y, c=objective_values)
        plt.colorbar()
        plt.show()

    # Find the shift that produced the highest score
    best_shift = shifts[np.argmax(objective_values)]
    return best_shift


def obj_mee_matching(
    shifted_field_trees: gpd.GeoDataFrame,
    drone_trees: gpd.GeoDataFrame,
    obs_bounds: gpd.GeoDataFrame,
    min_height: float = 10,
    edge_buffer: float = 5,
    height_column: str = "height",
) -> float:
    """
    Compute the F1 score for how many trees matched.
    Adapted from: https://github.com/open-forest-observatory/ofo-r/blob/3e3d138ffd99539affb7158979d06fc535bc1066/R/tree-map-alignment.R#L138


    Args:
        shifted_field_trees (gpd.GeoDataFrame): The field trees with a candidate shift applied
        drone_trees (gpd.GeoDataFrame): The drone trees
        obs_bounds (gpd.GeoDataFrame): The region that was surveyed, shifted commensurately with the field trees
        min_height (float, optional): Minimum height of field trees to evaluate. Defaults to 10.
        edge_buffer (float, optional): Only score trees this distance from the boundary of the survey region. Defaults to 5.
        height_column (str, optional): What column represents the tree heights. Defaults to "height".

    Returns:
        float: The F1 score for matching
    """
    # Crop to the observation bounds
    shifted_field_trees_cropped = shifted_field_trees.clip(
        obs_bounds.geometry.values[0]
    )
    drone_trees_cropped = drone_trees.clip(obs_bounds.geometry.values[0])
    # Reset the index to integers starting at 0
    shifted_field_trees_cropped.reset_index(inplace=True, drop=True)
    drone_trees_cropped.reset_index(inplace=True, drop=True)

    # Compute the matches between the shifted field points and the drone points
    matched_field_tree_inds, matched_drone_tree_inds = match_trees_singlestratum(
        field_trees=shifted_field_trees_cropped,
        drone_trees=drone_trees_cropped,
        vis=False,
    )

    # From the tree inds, we get the number of matches. Now all that's left to do is compute which
    # fraction of those fall within the core area and how many remain

    obs_bounds_core = obs_bounds.geometry.values[0].buffer(-edge_buffer)

    # Crop to the central area to avoid edge effects
    core_field_trees = shifted_field_trees_cropped.clip(obs_bounds_core)
    core_drone_trees = drone_trees_cropped.clip(obs_bounds_core)
    # Remove the short trees
    core_field_trees = core_field_trees[core_field_trees[height_column] >= min_height]
    core_drone_trees = core_drone_trees[core_drone_trees[height_column] >= min_height]

    # Find which of the core trees were matched based on their indices
    field_core_matched = set(matched_field_tree_inds).intersection(
        set(core_field_trees.index)
    )
    drone_core_matched = set(matched_drone_tree_inds).intersection(
        set(core_drone_trees.index)
    )

    # Compute precision and recall, using only the core trees as the denominator
    recall = (
        len(field_core_matched) / len(core_field_trees)
        if len(core_field_trees) > 0
        else 0
    )
    precision = (
        len(drone_core_matched) / len(core_drone_trees)
        if len(core_drone_trees) > 0
        else 0
    )

    # Compute the F1 score, setting to 0 if both precision and recall are 0
    f1 = (
        (2 * (precision * recall) / (precision + recall))
        if (precision + recall) > 0
        else 0
    )

    return f1


def match_trees_singlestratum(
    field_trees,
    drone_trees,
    search_height_proportion=0.5,
    search_distance_fun_slope=0.1,
    search_distance_fun_intercept=1,
    height_col="height",
    vis=False,
):
    # A reimplementation of
    # https://github.com/open-forest-observatory/ofo-r/blob/3e3d138ffd99539affb7158979d06fc535bc1066/R/tree-detection-accuracy-assessment.R#L164
    # Compute the pairwise distance matrix (dense, I don't see a way around it)
    field_tree_points_np = shapely.get_coordinates(field_trees.geometry)
    drone_tree_points_np = shapely.get_coordinates(drone_trees.geometry)

    # consider if this order should be switched
    # This looks correct, it seems like the observed trees are vertical
    distance_matrix = cdist(field_tree_points_np, drone_tree_points_np)

    # Expand so the field trees are a tall matrix and the drone trees are a wide one
    field_height = np.expand_dims(field_trees[height_col].to_numpy(), axis=1)
    drone_height = np.expand_dims(drone_trees[height_col].to_numpy(), axis=0)

    # Compute upper and lower height bounds for matches
    min_drone_height = field_height * (1 - search_height_proportion)
    max_drone_height = field_height * (1 + search_height_proportion)
    # Compute max spatial distances for valid matches
    max_dist = field_height * search_distance_fun_slope + search_distance_fun_intercept

    # Compute which matches fit the criteria using broadcasting to get a matrix representation
    above_min_height = drone_height > min_drone_height
    below_max_height = drone_height < max_drone_height
    below_max_matching_dist = distance_matrix < max_dist

    # Compute which matches fit all three criteria
    possible_pairings = np.logical_and.reduce(
        [above_min_height, below_max_height, below_max_matching_dist]
    )

    # Extract the indices of possible pairings
    possible_pairing_field_inds, possible_paring_drone_inds = np.where(
        possible_pairings
    )
    possible_pairing_inds = np.vstack(
        [possible_pairing_field_inds, possible_paring_drone_inds]
    ).T

    # Extract the distances corresponding to the valid matches
    possible_dists = distance_matrix[
        possible_pairing_field_inds, possible_paring_drone_inds
    ]

    # Sort so the paired indices are sorted, corresponding to the smallest distance pair first
    ordered_by_dist = np.argsort(possible_dists)
    possible_pairing_inds = possible_pairing_inds[ordered_by_dist]

    # Compute the most possible pairs, which is the min of num field and drone trees
    max_valid_matches = np.min(distance_matrix.shape)

    # Record the valid mathces
    matched_field_tree_inds = []
    matched_drone_tree_inds = []

    # Iterate over the indices
    for field_ind, drone_ind in possible_pairing_inds:
        # If neither the field or drone tree has already been matched, this is a valid pairing
        if (field_ind not in matched_field_tree_inds) and (
            drone_ind not in matched_drone_tree_inds
        ):
            # Add the matches to the lists
            matched_field_tree_inds.append(field_ind)
            matched_drone_tree_inds.append(drone_ind)

        # Check to see if all possible trees have been matched. Note, the length of matched field
        # and matched drone inds is the same, so we only need to check one.
        if len(matched_field_tree_inds) == max_valid_matches:
            break

    if vis:
        # Visualize matches
        f, ax = plt.subplots()
        ax.scatter(x=field_tree_points_np[:, 0], y=field_tree_points_np[:, 1], c="r")
        ax.scatter(x=drone_tree_points_np[:, 0], y=drone_tree_points_np[:, 1], c="b")

        ordered_matched_field_trees = field_tree_points_np[matched_field_tree_inds]
        ordered_matched_drone_trees = drone_tree_points_np[matched_drone_tree_inds]
        lines = [
            [tuple(x), tuple(y)]
            for x, y in zip(ordered_matched_field_trees, ordered_matched_drone_trees)
        ]

        lc = mc.LineCollection(lines, colors="k", linewidths=2)
        ax.add_collection(lc)

        plt.show()
    return matched_field_tree_inds, matched_drone_tree_inds


def align_plot(field_trees, drone_trees, obs_bounds, height_column="height", vis=False):
    original_field_CRS = field_trees.crs
    # Transform the drone trees to a cartesian CRS if not already
    field_trees = ensure_projected_CRS(field_trees)

    # Ensure that drone trees are in the same CRS
    drone_trees.to_crs(field_trees.crs, inplace=True)
    obs_bounds.to_crs(field_trees.crs, inplace=True)

    # First compute a rough shift and then a fine one
    coarse_shift = find_best_shift(
        field_trees=field_trees,
        drone_trees=drone_trees,
        obs_bounds=obs_bounds,
        objective_function=obj_mee_matching,
        search_increment=1,
        search_window=10,
        vis=True,
    )
    # This is initialized from the coarse shift
    fine_shift = find_best_shift(
        field_trees=field_trees,
        drone_trees=drone_trees,
        obs_bounds=obs_bounds,
        objective_function=obj_mee_matching,
        search_window=2,
        search_increment=0.2,
        base_shift=coarse_shift,
    )

    print(f"Rough shift: {coarse_shift}, fine shift: {fine_shift}")

    shifted_field_trees = field_trees.copy()
    # Apply the computed shift to the geometry of all field trees
    shifted_field_trees.geometry = shifted_field_trees.geometry.apply(
        lambda x: translate(x, xoff=fine_shift[0], yoff=fine_shift[1])
    )

    # Convert back to the original CRS
    shifted_field_trees.to_crs(original_field_CRS, inplace=True)

    if vis:
        # Plot the aligned data
        f, ax = plt.subplots()
        shifted_field_trees.plot(ax=ax)
        drone_trees.plot(ax=ax)
        plt.show()

    return shifted_field_trees, fine_shift


def match_field_and_drone_trees(
    field_trees: gpd.GeoDataFrame,
    drone_trees: gpd.GeoDataFrame,
    drone_crowns: gpd.GeoDataFrame,
    field_perim: gpd.GeoDataFrame,
    field_buffer_dist: float = 10.0,
) -> gpd.GeoDataFrame:
    """
    Add information from the field points to the drone crowns, using the corresponding tree tops
    to link the two.

    Args:
        field_trees (gpd.GeoDataFrame): The field surveyed trees
        drone_trees (gpd.GeoDataFrame): The detected tree tops
        drone_crowns (gpd.GeoDataFrame): The detected crowns
        field_perim (gpd.GeoDataFrame): GeoDataFrame with a single polygon geometry representing the surveyed area corresponding
        field_buffer_dist (float, optional): Consider matching to drone trees within this distance of the field trees. Defaults to 10.0.

    Returns:
        gpd.GeoDataFrame: Drone crowns with additional attributes from the field survey
    """
    # Ensure they are all in the same projected CRS
    field_trees = ensure_projected_CRS(field_trees)
    drone_trees = drone_trees.to_crs(field_trees.crs)
    drone_crowns = drone_crowns.to_crs(field_trees.crs)
    field_perim = field_perim.to_crs(field_trees.crs)

    # Get the buffered perimiter
    perim_buff = field_perim.buffer(field_buffer_dist).geometry.values[0]

    # Consider within vs intersects or other options
    drone_trees = drone_trees[drone_trees.within(perim_buff)]
    drone_trees.index = np.arange(len(drone_trees))

    # Maybe filter some of the short trees
    # Compute the full distance matrix or at least the top n matches
    matched_field_tree_inds, matched_drone_tree_inds = match_trees_singlestratum(
        field_trees=field_trees, drone_trees=drone_trees, vis=False
    )

    # Compute field trees that were matched
    matched_field_trees = field_trees.iloc[matched_field_tree_inds]
    # Drop the geometry from the field trees since we don't want to keep it
    matched_field_trees.drop("geometry", axis=1, inplace=True)
    # Compute the "unique_ID" for matched drone trees. This is a crosswalk with the
    # "treetop_unique_ID" field in the crown polygons
    drone_tree_unique_IDs = drone_trees.iloc[
        matched_drone_tree_inds
    ].unique_ID.to_numpy()
    # These two variables, matched_field_trees and drone_tree_unique_IDs, are now ordered in the same way
    # This means corresponding rows should be paired. Effectively, we could add the
    # drone_tree_unique_ID as a column of the field trees and then merge based on that. But we don't
    # want to modify the dataframe, so it's just provided for the `merge` step.

    # Transfer the attributes to the drone trees.
    drone_crowns_with_additional_attributes = pd.merge(
        left=drone_crowns,
        right=matched_field_trees,
        left_on="treetop_unique_ID",
        right_on=drone_tree_unique_IDs,
        how="left",
        suffixes=(
            "_drone",
            "_field",
        ),  # Append these suffixes in cases of name collisions
    )

    return drone_crowns_with_additional_attributes


if __name__ == "__main__":
    FIELD_REF = "/ofo-share/repos-david/geospatial-data-registration-toolkit/data/points/0002_field_trees.gpkg"
    DETECTED_TREES = "/ofo-share/repos-david/geospatial-data-registration-toolkit/data/points/0002_000451_000446_detected.gpkg"
    PLOT_BOUNDS = "/ofo-share/repos-david/geospatial-data-registration-toolkit/data/points/0002_plot.gpkg"

    field_trees = gpd.read_file(FIELD_REF)
    detected_trees = gpd.read_file(DETECTED_TREES)
    plot_bounds = gpd.read_file(PLOT_BOUNDS)

    align_plot(field_trees, detected_trees, obs_bounds=plot_bounds)
