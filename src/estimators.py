from sklearn.linear_model import Ridge
import numpy as np

"""
In this file we collect different methods that solve the following task:

Given a (complete) matrix S, and the values x, and q of the anchor rows/columns,
predict the value of the missing entry (i,j).
"""


class BaseEstimator():
    def predict(self, S, x, q):
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


class SNNEstimator(BaseEstimator):
    def universal_rank(s, m, n):
        """
        retain all singular values above optimal threshold as per Donoho & Gavish '14:
        https://arxiv.org/pdf/1305.5870.pdf
        """
        ratio = m / n
        omega = 0.56*ratio**3 - 0.95*ratio**2 + 1.43 + 1.82*ratio
        t = omega * np.median(s)
        rank = max(len(s[s > t]), 1)
        return rank

    def __init__(self, spectral_rank_fun=universal_rank):
        """
        spectral_rank_fun: function (sing_vals, m, n -> number),
                    determines how many singular values to keep.
        """
        self.calc_rank = spectral_rank_fun

    def predict(self, S, x, q):
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
        reg_alpha: float or function (size:number -> float),
                    regularization strength for the ridge regression.
        """
        self.reg_alpha = reg_alpha

    def predict(self, S, x, q):
        alpha = self.reg_alpha(len(q.flatten())) if callable(
            self.reg_alpha) else self.reg_alpha
        model = Ridge(alpha=alpha)
        model.fit(S.T, q)
        est = model.predict(x.reshape(1, -1))
        return est
