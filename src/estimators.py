from sklearn.linear_model import Ridge
import numpy as np

"""
In this file we collect different methods that solve the following task:

Given a (complete) matrix S, and the values x, and q of the anchor rows/columns,
predict the value of the missing entry (i,j).
"""


class BaseEstimator():
    def predict(self, S, x, q, ind_row, ind_col):
        """
        Given the following setting, predict the entry (i,j):
          [ ... q ... ]  [(i,j)]

          [    ...    ]  [.]
          [ ... S ... ]  [x]
          [    ...    ]  [.]

        S: a (complete) matrix of shape (n, m)
        q: row of shape (1, m)
        x: column of shape (n, 1)
        """
        return 0.0

    def prepare(self, M, D):
        """Override if the estimator needs M and D."""
        pass


class SNNEstimator(BaseEstimator):
    def universal_rank(s, m, n):
        """
        retain all singular values above optimal threshold as per Donoho & Gavish '14:
        https: // arxiv.org/pdf/1305.5870.pdf
        """
        ratio = m / n
        omega = 0.56*ratio**3 - 0.95*ratio**2 + 1.43 + 1.82*ratio
        t = omega * np.median(s)
        rank = max(len(s[s > t]), 1)
        return rank

    def __init__(self, spectral_rank_fun=universal_rank):
        """
        spectral_rank_fun: function(sing_vals, m, n -> number),
                    determines how many singular values to keep.
        """
        self.calc_rank = spectral_rank_fun

    def predict(self, S, x, q, ind_row, ind_col):
        u, s, v = np.linalg.svd(S, full_matrices=False)
        # keep at least one singular value
        rank = max(1, self.calc_rank(s, *S.shape))
        rank = min(rank, len(s))
        svd_inv_T = np.sum([1/s[l] * u[:, [l]] * v[[l], :]
                            for l in range(rank)], axis=0)
        beta = svd_inv_T @ q.reshape(-1, 1)
        est = np.dot(beta.flatten(), x)
        return est


class RidgeEstimator(BaseEstimator):
    def __init__(self, reg_alpha=lambda sz: sz*0.01):
        """
        reg_alpha: float or function(size: number -> float),
                    regularization strength for the ridge regression.
        """
        self.reg_alpha = reg_alpha

    def predict(self, S, x, q, ind_row, ind_col):
        alpha = self.reg_alpha(len(q.flatten())) if callable(
            self.reg_alpha) else self.reg_alpha
        model = Ridge(alpha=alpha)
        model.fit(S.T, q)
        est = model.predict(x.reshape(1, -1))
        return est


class GapEstimator(BaseEstimator):
    def __init__(self, avg_method="row", estimator=RidgeEstimator()):
        """
        Can predict a value with non-complete submatrix S.
        For that, it fills missing values in S with row or column averages.
        M: complete matrix,
        D: binary matrix indicating missing values.
        avg_method: how to impute missing values. Available options:
            "row", "column"
        estimator: estimator to use after imputing missing values.
        """
        self.estimator = estimator
        self.avg_method = avg_method

    def prepare(self, M, D):
        M_complete = M.copy()
        # fill missing values with row/column averages
        if self.avg_method == "row":
            for row in range(M.shape[0]):
                row_avg = np.mean(M[row, :][D[row, :] == 1])
                M_complete[row, :][D[row, :] == 0] = row_avg
        elif self.avg_method == "column":
            for col in range(M.shape[1]):
                col_avg = np.mean(M[:, col][D[:, col] == 1])
                M_complete[:, col][D[:, col] == 0] = col_avg
        else:
            raise ValueError(
                "avg_method must be one of 'row', 'column'")
        assert(np.all(np.isfinite(M_complete)))
        self.M_complete = M_complete

    def predict(self, S, x, q, ind_row, ind_col):
        # S is may not be complete, so we need to fill missing values
        S_complete = S.copy()
        mask = np.isnan(S)
        S_complete[mask] = self.M_complete[np.ix_(ind_row, ind_col)][mask]
        assert(np.all(np.isfinite(S_complete)))
        return self.estimator.predict(S_complete, x, q, ind_row, ind_col)
