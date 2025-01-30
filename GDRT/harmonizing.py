from itertools import chain
import numpy as np
import scipy
from typing import List, Dict, Tuple

from IPython.core.debugger import set_trace


def compute_global_shifts_from_pairwise(
    shifts: Dict[Tuple[str, str], Tuple[float, float]],
    shift_weights=List[float],
    dataset_weights=Dict[str, float],
) -> Dict[str, list[float, float]]:
    """_summary_

    Args:
        shifts (Dict[Tuple[str, str], Tuple[float, float]]): The pairwise shifts between different datasets. The keys are 2-tuples
        consisting of two dataset IDs for which a relative shift has been identified. The values
        are the (x, y) values of the shift. The second dataset should be shifted by the identified
        amount to match the first one.
        shift_weights (List[float]): How much important to give each of these shifts in the optimization.
        The shift_weights should be in the same order as the shifts and are used correspondingly.
        dataset_weights (Dict[str, float]): How much importance to give keeping each dataset the current location.

    Returns:
        Dict[str, List[float, float]]: The keys of this dictionary are the dataset ids and the
        values are the ammount that the datasets should be shifted by from their current position.
    """
    # Compute the unique dataset IDs across all the shift pairs
    unique_datasets = sorted(set(chain(*list(shifts.keys()))))

    n_shifts = len(shifts)
    n_datasets = len(unique_datasets)

    # the A matrix captures the x or y component of a pairwise shifts for each of the reported correspondences
    # Each row represents a pair, each column, a dataset
    A_shift = np.zeros((2 * n_shifts, 2 * n_datasets))
    b_shift = np.zeros((2 * n_shifts, 1))

    # Iterate over the shifts to populate the matrices
    for i, (datasets, xy_shift) in enumerate(shifts.items()):
        dataset_1_ind = unique_datasets.index(datasets[0])
        dataset_2_ind = unique_datasets.index(datasets[1])

        # This is the relative weight of this pair. It's multiplied by both sides of the equation.
        shift_weight = shift_weights[i]

        # The shift dataset 2 minus the shift of dataset 1 ideally should be the same as the identified shift
        # X shift
        A_shift[2 * i, 2 * dataset_1_ind] = -1 * shift_weight
        A_shift[2 * i, 2 * dataset_2_ind] = shift_weight
        b_shift[2 * i] = xy_shift[0] * shift_weight
        # Y shift
        A_shift[2 * i + 1, 2 * dataset_1_ind + 1] = -1 * shift_weight
        A_shift[2 * i + 1, 2 * dataset_2_ind + 1] = shift_weight
        b_shift[2 * i + 1] = xy_shift[1] * shift_weight

    A_absolute = np.zeros((2 * n_datasets, 2 * n_datasets))
    b_absolute = np.zeros((2 * n_datasets, 1))

    # Iterate over the dataset weights to constrain the absolute shifts
    for i, unique_dataset in enumerate(unique_datasets):
        dataset_weight = dataset_weights[unique_dataset]
        dataset_ind = unique_datasets.index(unique_dataset)

        # Constrain the x and y components
        A_absolute[2 * i, 2 * dataset_ind] = dataset_weight
        A_absolute[2 * i + 1, 2 * dataset_ind + 1] = dataset_weight

    A = np.concatenate((A_shift, A_absolute), axis=0)
    b = np.concatenate((b_shift, b_absolute), axis=0)

    print("Abotu to run lstqr")
    x, res, rank, s = scipy.linalg.lstsq(A, b)
    x = np.reshape(x, (-1, 2))
    absolute_shifts = {
        dataset_id: shift for dataset_id, shift in zip(unique_datasets, x)
    }
    return absolute_shifts
