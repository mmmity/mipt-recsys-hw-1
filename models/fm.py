"""FM второго порядка с ID и метаданными аниме.

Признаки: user ID, anime ID, жанры, тип и log(1 + episodes).
Обучаем логистическую FM на оценках 7–10 как положительных примерах,
оценках 1–6 и случайных непросмотренных аниме как отрицательных.
Просмотры с оценкой -1 не задают целевую метку.
Прогноз до sigmoid: b + w.T @ x + 0.5 * sum_f((sum_i V_if x_i)^2
- sum_i V_if^2 x_i^2). Минимизируем бинарную логистическую потерю.
"""

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import expit

from ._common import RecommenderBase


class FactorizationMachine(RecommenderBase):
    def __init__(
        self, factors=16, epochs=3, batch_size=4096, learning_rate=0.01,
        regularization=0.0001, negative_ratio=1, use_metadata=True, seed=1000 - 7,
    ):
        self.factors = factors
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.negative_ratio = negative_ratio
        self.use_metadata = use_metadata
        self.seed = seed

    def _build_item_features(self, anime):
        if not self.use_metadata:
            self.item_features = sparse.eye(len(self.item_ids), format="csr")
            self.n_features = len(self.user_ids) + len(self.item_ids)
            return
        metadata = anime.drop_duplicates("anime_id").set_index("anime_id").reindex(self.item_ids)
        # создаем разреженные признаки аниме из доступных метаданных
        genre_lists = [
            [part.strip() for part in value.split(",") if part.strip()]
            for value in metadata.genre
        ]
        genres = sorted({genre for values in genre_lists for genre in values})
        types = sorted(metadata.type.unique())
        genre_index = {genre: i for i, genre in enumerate(genres)}
        type_index = {value: i for i, value in enumerate(types)}
        self.genres = genres
        self.types = types

        n_items = len(self.item_ids)
        item_identity = sparse.eye(n_items, format="csr")
        genre_rows = []
        genre_cols = []
        genre_values = []
        for row, values in enumerate(genre_lists):
            for genre in values:
                genre_rows.append(row)
                genre_cols.append(genre_index[genre])
                genre_values.append(1 / np.sqrt(len(values)))
        genre_features = sparse.csr_matrix(
            (genre_values, (genre_rows, genre_cols)), shape=(n_items, len(genres))
        )
        type_features = sparse.csr_matrix(
            (
                np.ones(n_items),
                (np.arange(n_items), [type_index[value] for value in metadata.type]),
            ),
            shape=(n_items, len(types)),
        )
        episodes = np.log1p(pd.to_numeric(metadata.episodes_num).to_numpy())
        episode_std = episodes.std()
        if episode_std > 0:
            episodes = (episodes - episodes.mean()) / episode_std
        else:
            episodes = np.zeros(n_items)
        episode_features = sparse.csr_matrix(episodes.reshape(-1, 1))
        self.item_features = sparse.hstack(
            [item_identity, genre_features, type_features, episode_features], format="csr"
        )
        self.n_features = len(self.user_ids) + self.item_features.shape[1]

    def _design_matrix(self, user_indices, item_indices):
        n_rows = len(user_indices)
        user_features = sparse.csr_matrix(
            (np.ones(n_rows), (np.arange(n_rows), user_indices)),
            shape=(n_rows, len(self.user_ids)),
        )
        return sparse.hstack(
            [user_features, self.item_features[item_indices]], format="csr"
        )

    def _sample_unseen(self, positive_users, known_rows, known_cols, n_items, rng):
        users = np.repeat(positive_users, self.negative_ratio)
        known_count = np.bincount(known_rows, minlength=len(self.user_ids))
        users = users[known_count[users] < n_items]
        items = rng.integers(n_items, size=len(users))
        known_pairs = known_rows * n_items + known_cols
        bad = np.isin(users * n_items + items, known_pairs)
        while bad.any():
            items[bad] = rng.integers(n_items, size=bad.sum())
            bad = np.isin(users * n_items + items, known_pairs)
        return users, items

    def fit(self, interactions, anime):
        rows, cols, shape = self._prepare(interactions)
        self._build_item_features(anime)
        sample_seed, initialization_seed, batch_seed = np.random.SeedSequence(self.seed).spawn(3)
        sample_rng = np.random.default_rng(sample_seed)
        initialization_rng = np.random.default_rng(initialization_seed)
        batch_rng = np.random.default_rng(batch_seed)

        # формируем одинаковые обучающие примеры для всех эпох
        ratings = interactions.rating.to_numpy()
        positive = ratings >= 7
        negative = (ratings >= 1) & (ratings < 7)
        sampled_users, sampled_items = self._sample_unseen(
            rows[positive], rows, cols, shape[1], sample_rng
        )
        users = np.concatenate([rows[positive], rows[negative], sampled_users])
        items = np.concatenate([cols[positive], cols[negative], sampled_items])
        labels = np.concatenate([
            np.ones(positive.sum()), np.zeros(negative.sum() + len(sampled_users))
        ])
        features = self._design_matrix(users, items)

        self.bias = 0.0
        self.linear = np.zeros(self.n_features)
        self.latent = initialization_rng.normal(0, 0.01, (self.n_features, self.factors))
        linear_mean = np.zeros_like(self.linear)
        linear_variance = np.zeros_like(self.linear)
        latent_mean = np.zeros_like(self.latent)
        latent_variance = np.zeros_like(self.latent)
        step = 0

        # обновляем параметры логистической FM мини-батчами Адамом
        for _ in range(self.epochs):
            order = batch_rng.permutation(len(labels))
            for start in range(0, len(labels), self.batch_size):
                batch = order[start:start + self.batch_size]
                x = features[batch]
                x_squared = x.multiply(x)
                y = labels[batch]
                interactions_sum = x @ self.latent
                interaction_terms = x_squared @ (self.latent * self.latent)
                logits = (
                    self.bias + x @ self.linear
                    + 0.5 * np.sum(interactions_sum**2 - interaction_terms, axis=1)
                )
                error = expit(logits) - y
                linear_gradient = (
                    np.asarray(x.T @ error).ravel() / len(batch)
                    + self.regularization * self.linear
                )
                latent_gradient = (
                    np.asarray(x.T @ (error[:, None] * interactions_sum)) / len(batch)
                    - np.asarray(x_squared.T @ error).reshape(-1, 1) * self.latent / len(batch)
                    + self.regularization * self.latent
                )

                step += 1
                linear_mean = 0.9 * linear_mean + 0.1 * linear_gradient
                linear_variance = 0.999 * linear_variance + 0.001 * linear_gradient**2
                latent_mean = 0.9 * latent_mean + 0.1 * latent_gradient
                latent_variance = 0.999 * latent_variance + 0.001 * latent_gradient**2
                linear_adjusted = linear_mean / (1 - 0.9**step)
                linear_scale = linear_variance / (1 - 0.999**step)
                latent_adjusted = latent_mean / (1 - 0.9**step)
                latent_scale = latent_variance / (1 - 0.999**step)
                self.linear -= self.learning_rate * linear_adjusted / (np.sqrt(linear_scale) + 0.00000001)
                self.latent -= self.learning_rate * latent_adjusted / (np.sqrt(latent_scale) + 0.00000001)
                self.bias -= self.learning_rate * error.mean()
        return self

    def _score_known(self, user_index, item_indices):
        users = np.full(len(item_indices), user_index)
        features = self._design_matrix(users, item_indices)
        interactions_sum = features @ self.latent
        interaction_terms = features.multiply(features) @ (self.latent * self.latent)
        return np.asarray(
            self.bias + features @ self.linear
            + 0.5 * np.sum(interactions_sum**2 - interaction_terms, axis=1)
        ).ravel()
