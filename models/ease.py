"""EASE: линейная модель с нулевой диагональю матрицы весов.

P = (X.T @ X + regularization * I)^-1, B_ij = -P_ij / P_jj.
Диагональ B равна нулю, прогноз равен X_u @ B.
"""

import numpy as np
from scipy import sparse

from ._common import RecommenderBase


class EASE(RecommenderBase):
    def __init__(self, regularization=200):
        self.regularization = regularization

    def fit(self, interactions, anime=None):
        rows, cols, shape = self._prepare(interactions)

        # обучаем линейные веса на матрице положительных взаимодействий
        liked = interactions.rating.to_numpy() >= 7
        self.likes = sparse.csr_matrix(
            (np.ones(liked.sum()), (rows[liked], cols[liked])), shape=shape
        )
        gram = (self.likes.T @ self.likes).toarray()
        gram.flat[::shape[1] + 1] += self.regularization
        inverse = np.linalg.inv(gram)
        self.weights = -inverse / np.diag(inverse)[None, :]
        np.fill_diagonal(self.weights, 0)
        return self

    def _score_known(self, user_index, item_indices):
        if self.likes[user_index].nnz == 0:
            return self.popularity[item_indices]
        return np.asarray(self.likes[user_index] @ self.weights[:, item_indices]).ravel()
