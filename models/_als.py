"""Общее ALS-решение для двух разных целевых сигналов."""

import numpy as np
from scipy import sparse

from ._common import RecommenderBase


class ALSBase(RecommenderBase):
    def __init__(self, factors=32, iterations=10, alpha=20, regularization=0.1, seed=1000 - 7):
        self.factors = factors
        self.iterations = iterations
        self.alpha = alpha
        self.regularization = regularization
        self.seed = seed

    @staticmethod
    def _update(other, confidence_extra, weighted_target, regularization):
        n_rows = confidence_extra.shape[0]
        factors = np.zeros((n_rows, other.shape[1]))
        base_gram = other.T @ other
        identity = np.eye(other.shape[1])

        for row in range(n_rows):
            start, end = confidence_extra.indptr[row:row + 2]
            if start == end:
                continue
            indices = confidence_extra.indices[start:end]
            extra = confidence_extra.data[start:end]
            target = weighted_target.data[start:end]
            selected = other[indices]
            gram = base_gram + selected.T @ (extra[:, None] * selected)
            rhs = target @ selected
            factors[row] = np.linalg.solve(gram + regularization * identity, rhs)
        return factors

    def _fit_als(self, rows, cols, shape, confidence_extra, weighted_target):
        # уверенность и целевой сигнал задаем только для наблюдаемых пар
        extra = sparse.csr_matrix((confidence_extra, (rows, cols)), shape=shape)
        target = sparse.csr_matrix((weighted_target, (rows, cols)), shape=shape)
        rng = np.random.default_rng(self.seed)
        self.user_factors = rng.normal(0, 0.01, (shape[0], self.factors))
        self.item_factors = rng.normal(0, 0.01, (shape[1], self.factors))

        # чередуем точные решения для факторов пользователей и аниме
        item_extra = extra.T.tocsr()
        item_target = target.T.tocsr()
        for _ in range(self.iterations):
            self.user_factors = self._update(
                self.item_factors, extra, target, self.regularization
            )
            self.item_factors = self._update(
                self.user_factors, item_extra, item_target, self.regularization
            )
        return self

    def _score_known(self, user_index, item_indices):
        return self.item_factors[item_indices] @ self.user_factors[user_index]
