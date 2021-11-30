import numpy as np
import src.anchor_matrix as anchor_matrix
from src.estimators import SNNEstimator, BaseEstimator


def general_snn_ij(D, Y, i, j, num_estimates, biclique_search, estimator: BaseEstimator):
    estimates = []
    while len(estimates) < num_estimates:
        ind_row, ind_col = anchor_matrix.anchor_sub_matrix(
            D, i, j, biclique_search=biclique_search)
        x, q = Y[ind_row, j], Y[i, ind_col]
        S = Y[np.ix_(ind_row, ind_col)]  # create submatrix
        est = estimator.predict(S, x, q)
        estimates.append(est)
    return np.mean(estimates)


def general_snn(
    D,
    Y,
    biclique_search=anchor_matrix.biclique_find,
    estimator=SNNEstimator(),
    num_estimates=1,
    min_val=None,
    max_val=None,
    print_progress=False,
):
    '''
    D: boolean matrix (shape (N, M)), True if value Y[i,j] is observed
    Y: float matrix (shape (N, M))
    '''
    A = Y.copy()
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if not D[i, j]:
                A[i, j] = general_snn_ij(
                    D, Y, i, j, num_estimates, biclique_search, estimator)
        if print_progress:
            print(f'{i}/{A.shape[0]}')

    # clipping
    if min_val is not None:
        A[A < min_val] = min_val
    if max_val is not None:
        A[A > max_val] = max_val
    return A
