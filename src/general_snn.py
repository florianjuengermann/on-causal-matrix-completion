import numpy as np
import src.anchor_matrix as anchor_matrix
from src.estimators import SNNEstimator, BaseEstimator


def general_snn_ij(D, Y, i, j, num_estimates, biclique_search, estimator: BaseEstimator, multiest = []):
    if len(multiest) == 0:
        estimates = []
        while len(estimates) < num_estimates:
            ind_row, ind_col = anchor_matrix.anchor_sub_matrix(
                D, i, j, biclique_search=biclique_search)
            x, q = Y[ind_row, j], Y[i, ind_col]
            S = Y[np.ix_(ind_row, ind_col)]  # create submatrix
            est = estimator.predict(S, x, q)
            estimates.append(est)
        return np.mean(estimates)
    
    else:
        estimates = {}
        
        for est in multiest:
            estimates[est.__name__] = []
        
        c = 0
        while c < num_estimates:
            ind_row, ind_col = anchor_matrix.anchor_sub_matrix(
                D, i, j, biclique_search=biclique_search)
            x, q = Y[ind_row, j], Y[i, ind_col]
            S = Y[np.ix_(ind_row, ind_col)]  # create submatrix
            for est in multiest:
                pred = est.predict(S, x, q)
                estimates[est.__name__].append(pred)
            c += 1
        
        for est in multiest:
            estimates[est.__name__] = np.mean(estimates[est.__name__])
        
        return estimates


def general_snn(
    D,
    Y,
    biclique_search=anchor_matrix.biclique_find,
    estimator=SNNEstimator(),
    multiest = [],
    num_estimates=1,
    min_val=None,
    max_val=None,
    print_progress=False,
):
    '''
    D: boolean matrix (shape (N, M)), True if value Y[i,j] is observed
    Y: float matrix (shape (N, M))
    '''
    if len(multiest) == 0:
        A = Y.copy()
        for i in range(A.shape[0]):
            if print_progress:
                print("\r", f'{i}/{A.shape[0]}', end="")
            for j in range(A.shape[1]):
                if not D[i, j]:
                    A[i, j] = general_snn_ij(
                        D, Y, i, j, num_estimates, biclique_search, estimator, multiest)
        if print_progress:
            print()

        # clipping
        if min_val is not None:
            A[A < min_val] = min_val
        if max_val is not None:
            A[A > max_val] = max_val
        return A
    
    else:
        A = {}
        
        for est in multiest:
            A[est.__name__] = Y.copy()
        
        for i in range(Y.shape[0]):
            if print_progress:
                print("\r", f'{i}/{Y.shape[0]}', end="")
            for j in range(Y.shape[1]):
                if not D[i, j]:
                    results = general_snn_ij(
                        D, Y, i, j, num_estimates, biclique_search, estimator, multiest)
                    for est in multiest:
                        A[est.__name__][i,j] = results[est.__name__]
        if print_progress:
            print()

        # clipping
        for est in multiest:
            if min_val is not None:
                A[est.__name__][A[est.__name__] < min_val] = min_val
            if max_val is not None:
                A[est.__name__][A[est.__name__] > max_val] = max_val
        return A


