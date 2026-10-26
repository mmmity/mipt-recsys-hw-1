"""Общий интерфейс для пяти рекомендательных моделей."""

import numpy as np
from scipy import sparse


class RecommenderBase:
    def _prepare(self, interactions):
        self.user_ids = np.sort(interactions.user_id.unique())
        self.item_ids = np.sort(interactions.anime_id.unique())
        self.user_to_index = {user_id: i for i, user_id in enumerate(self.user_ids)}
        self.item_to_index = {item_id: i for i, item_id in enumerate(self.item_ids)}

        rows = interactions.user_id.map(self.user_to_index).to_numpy()
        cols = interactions.anime_id.map(self.item_to_index).to_numpy()
        shape = (len(self.user_ids), len(self.item_ids))
        self.seen = sparse.csr_matrix((np.ones(len(rows), dtype=bool), (rows, cols)), shape=shape)

        relevant = (interactions.rating.to_numpy() >= 7).astype(float)
        self.popularity = np.bincount(cols, weights=relevant, minlength=shape[1])
        if self.popularity.max() > 0:
            self.popularity = self.popularity / self.popularity.max()
        return rows, cols, shape

    def score(self, user_id, item_ids):
        item_ids = np.asarray(item_ids)
        result = np.full(item_ids.shape, -np.inf, dtype=float)
        known = np.array([item_id in self.item_to_index for item_id in item_ids])
        if not known.any():
            return result

        item_indices = np.array([self.item_to_index[item_id] for item_id in item_ids[known]])
        user_index = self.user_to_index.get(user_id)
        if user_index is None:
            result[known] = self.popularity[item_indices]
        else:
            result[known] = self._score_known(user_index, item_indices)
        return result

    def recommend(self, user_id, k=10):
        if k <= 0:
            return np.array([], dtype=self.item_ids.dtype)
        user_index = self.user_to_index.get(user_id)
        if user_index is None:
            candidates = self.item_ids
        else:
            known_items = self.seen[user_index].indices
            candidates = np.delete(self.item_ids, known_items)
        scores = self.score(user_id, candidates)
        return candidates[np.argsort(-scores, kind="stable")[:k]]
