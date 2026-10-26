"""iALS с собственным целевым сигналом из пользовательской оценки.

Для оценки r от 1 до 10 задаем t_ui = max(0, (r - 6) / 4).
Таким образом 1–6 дают 0, а 7–10 дают от 0.25 до 1.
Для оцененных пар c_ui = 1 + alpha. Просмотр без оценки r = -1
оставляем с базовой уверенностью c_ui = 1 и целью 0, поэтому он
не сообщает модели о предпочтении. В базовом iALS его цель равна 1.
Меняется именно t_ui, а не только веса уверенности.
"""

import numpy as np

from ._als import ALSBase


class RatingALS(ALSBase):
    def fit(self, interactions, anime=None):
        rows, cols, shape = self._prepare(interactions)
        ratings = interactions.rating.to_numpy()
        rated = ratings >= 1
        target = np.where(rated, np.maximum(0, (ratings - 6) / 4), 0)
        extra = self.alpha * rated.astype(float)
        weighted_target = (1 + extra) * target
        self.target_formula = "max(0, (rating - 6) / 4) for rated interactions"
        return self._fit_als(rows, cols, shape, extra, weighted_target)
