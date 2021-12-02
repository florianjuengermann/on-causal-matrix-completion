import numpy as np
from typing import List
import src.anchor_matrix as anchor_matrix
from src.estimators import SNNEstimator, BaseEstimator


def general_snn_multi_est_ij(D, Y, i, j, num_estimates,
                             biclique_search, estimators: List[BaseEstimator]):

    estimates = [[] for _ in estimators]
    while len(estimates[0]) < num_estimates:
        ind_row, ind_col = anchor_matrix.anchor_sub_matrix(
            D, i, j, biclique_search=biclique_search)
        x, q = Y[ind_row, j], Y[i, ind_col]
        S = Y[np.ix_(ind_row, ind_col)]  # create submatrix

        for est_id in range(len(estimators)):
            est = estimators[est_id].predict(S, x, q, ind_row, ind_col)
            estimates[est_id].append(est)
    return np.mean(estimates, axis=1)  # avg for each estimator


def general_snn_multi_est(
    D,
    Y,
    estimators: List[BaseEstimator],
    biclique_search=anchor_matrix.biclique_find,
    num_estimates=1,
    min_val=None,
    max_val=None,
    print_progress=False,
):

    # A is 3d matrix (M, N, len(estimators))
    A = np.repeat(Y[:, :, np.newaxis], len(estimators), axis=2)
    for i in range(A.shape[0]):
        if print_progress:
            print("\r", f'{i}/{A.shape[0]}', end="")
        for j in range(A.shape[1]):
            if not D[i, j]:
                A[i, j] = general_snn_multi_est_ij(
                    D, Y, i, j, num_estimates, biclique_search, estimators)
    if print_progress:
        print()

    # clipping
    if min_val is not None:
        A[A < min_val] = min_val
    if max_val is not None:
        A[A > max_val] = max_val
    # return list of matrices
    return np.moveaxis(A, 2, 0)


def general_snn(
    D,
    Y,
    estimator=SNNEstimator(),
    biclique_search=anchor_matrix.biclique_find,
    num_estimates=1,
    min_val=None,
    max_val=None,
    print_progress=False,
):
    '''
    D: boolean matrix (shape (N, M)), True if value Y[i,j] is observed
    Y: float matrix (shape (N, M))
    '''
    A_3d = general_snn_multi_est(D, Y, [estimator], biclique_search,
                                 num_estimates, min_val, max_val, print_progress)
    return A_3d[0]
