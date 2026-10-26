"""Базовый iALS: каждый наблюдаемый просмотр имеет целевой сигнал 1.

Минимизируем сумму c_ui (1 - x_u^T y_i)^2 для просмотров и
(x_u^T y_i)^2 для остальных пар с L2-регуляризацией.
Для просмотра c_ui = 1 + alpha.
"""

import numpy as np

from ._als import ALSBase


class ImplicitALS(ALSBase):
    def fit(self, interactions, anime=None):
        rows, cols, shape = self._prepare(interactions)
        extra = np.full(len(rows), self.alpha, dtype=float)
        weighted_target = np.full(len(rows), 1 + self.alpha, dtype=float)
        return self._fit_als(rows, cols, shape, extra, weighted_target)
