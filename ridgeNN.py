from sklearn.linear_model import Ridge
import numpy as np
import anchor_matrix


def RidgeNN_ij(D, Y, i, j):
    ind_row, ind_col = anchor_matrix.anchor_sub_matrix(D, i, j)
    x, q = Y[ind_row, j], Y[i, ind_col]
    S = Y[np.ix_(ind_row, ind_col)]  # create submatrix
    # S^T•β = q  ⇔  β = S^-T q
    model = Ridge(alpha=len(ind_row) * 0.01)
    model.fit(S.T, q)
    return model.predict(x.reshape(1, -1))


def RidgeNN(D, Y):
    '''
    D: boolean matrix (shape (N, M)), True if value Y[i,j] is observed
    Y: float matrix (shape (N, M))
    '''
    A = Y.copy()
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if not D[i, j]:
                A[i, j] = RidgeNN_ij(D, Y, i, j)
        #print(f"row {i}/{A.shape[0]}")
    return A
