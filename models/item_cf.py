"""Item-based CF с KNN по положительным взаимодействиям.

s_ij = cooc_ij / sqrt(pop_i pop_j) * cooc_ij / (cooc_ij + shrinkage).
Для каждого аниме оставляем neighbors ближайших по s_ij соседей.
Оценка объекта j для пользователя u равна сумме s_ij по его лайкам i.
"""

import numpy as np
from scipy import sparse

from ._common import RecommenderBase


class ItemCF(RecommenderBase):
    def __init__(self, neighbors=100, shrinkage=100):
        self.neighbors = neighbors
        self.shrinkage = shrinkage

    def fit(self, interactions, anime=None):
        rows, cols, shape = self._prepare(interactions)

        # строим матрицу понравившихся аниме и попарное сходство объектов
        liked = interactions.rating.to_numpy() >= 7
        self.likes = sparse.csr_matrix(
            (np.ones(liked.sum()), (rows[liked], cols[liked])), shape=shape
        )
        overlap = (self.likes.T @ self.likes).toarray()
        degree = np.diag(overlap).copy()
        normalizer = np.sqrt(np.outer(degree, degree))
        similarity = np.divide(overlap, normalizer, out=np.zeros_like(overlap), where=normalizer > 0)
        similarity *= np.divide(
            overlap, overlap + self.shrinkage,
            out=np.zeros_like(overlap), where=overlap + self.shrinkage > 0,
        )
        np.fill_diagonal(similarity, 0)

        # оставляем наиболее похожих соседей каждого аниме
        if self.neighbors < shape[1] - 1:
            selected = np.argpartition(similarity, -self.neighbors, axis=1)[:, -self.neighbors:]
            mask = np.zeros_like(similarity, dtype=bool)
            mask[np.arange(shape[1])[:, None], selected] = True
            similarity[~mask] = 0
        self.similarity = sparse.csr_matrix(similarity)
        return self

    def _score_known(self, user_index, item_indices):
        if self.likes[user_index].nnz == 0:
            return self.popularity[item_indices]
        scores = self.likes[user_index] @ self.similarity[:, item_indices]
        return np.asarray(scores.toarray()).ravel()
