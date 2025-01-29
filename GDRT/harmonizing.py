from itertools import chain
import numpy as np
import scipy

from IPython.core.debugger import set_trace


def compute_global_shifts_from_pairwise(
    shifts, shift_weights=None, current_location_weights=None
):
    """_summary_

    Args:
        shifts (_type_): The pairwise shifts between different datasets
        shift_weights (_type_): How much important to give each of these shifts in the optimization
        current_location_weights (_type_): How much importance to give the initial location of dataset
    """

    # Compute the unique dataset IDs across all the shift pairs
    unique_shift_keys = list(set(chain(*list(shifts.keys()))))

    n_shifts = len(shifts)
    n_unique_shifts = len(unique_shift_keys)
    print(unique_shift_keys)
    print(n_unique_shifts)

    # the A matrix captures the x or y component of a pairwise shifts for each of the reported correspondences
    # Each row represents a pair, each column, a dataset
    # TODO we may add one final row at the bottom to
    A = np.zeros((2 * n_shifts + 1, 2 * n_unique_shifts))
    b = np.zeros((2 * n_shifts + 1,))

    for i, (datasets, xy_shift) in enumerate(shifts.items()):
        dataset_1_ind = unique_shift_keys.index(datasets[0])
        dataset_2_ind = unique_shift_keys.index(datasets[1])

        # X shift
        # TODO ensure that the convention is correct here
        A[2 * i, 2 * dataset_1_ind] = 1
        A[2 * i, 2 * dataset_2_ind] = -1

        A[2 * i + 1, 2 * dataset_1_ind + 1] = 1
        A[2 * i + 1, 2 * dataset_2_ind + 1] = -1

        b[2 * i] = xy_shift[0]
        b[2 * i + 1] = xy_shift[1]

    # Minimize shifts
    A[-1, :] = 1
    print("Abotu to run lstqr")

    x, res, rank, s = scipy.linalg.lstsq(A, b)
    return x
