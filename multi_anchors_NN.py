from sklearn.linear_model import Ridge
import numpy as np
import anchor_matrix


def multi_anchors_NN_ij(D, Y, i, j, num_estimates):
    estimates = []
    while len(estimates) < num_estimates:
        ind_row, ind_col = anchor_matrix.anchor_sub_matrix(
            D, i, j, biclique_search=anchor_matrix.biclique_random)
        if len(ind_row) == 0 or len(ind_col) == 0:
            continue  # do not count as valid estimate
        x, q = Y[ind_row, j], Y[i, ind_col]
        S = Y[np.ix_(ind_row, ind_col)]  # create submatrix
        # S^T•β = q  ⇔  β = S^-T q
        model = Ridge(alpha=len(ind_row) * 0.01)
        model.fit(S.T, q)
        estimates.append(model.predict(x.reshape(1, -1)))
    return np.mean(estimates)


def multi_anchors_NN(D, Y, num_estimates=20):
    '''
    D: boolean matrix (shape (N, M)), True if value Y[i,j] is observed
    Y: float matrix (shape (N, M))
    '''
    A = Y.copy()
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if not D[i, j]:
                A[i, j] = multi_anchors_NN_ij(D, Y, i, j, num_estimates)
        print(f"row {i}/{A.shape[0]}")
    return A
