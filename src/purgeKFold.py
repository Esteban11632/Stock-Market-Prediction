import numpy as np

class PurgedKFoldCustom:
    def __init__(self, n_splits, samples_info_sets, pct_embargo=0.0):
        self.n_splits = n_splits
        self.samples_info_sets = samples_info_sets
        self.pct_embargo = pct_embargo

    def split(self, X):
        n_samples = len(X)

        indices = np.arange(n_samples)

        fold_sizes = np.full(
            self.n_splits,
            n_samples // self.n_splits,
            dtype=int
        )

        fold_sizes[: n_samples % self.n_splits] += 1

        current = 0

        for fold_size in fold_sizes:
            start = current
            stop = current + fold_size

            val_idx = indices[start:stop]

            val_start_time = self.samples_info_sets.index[val_idx[0]]
            val_end_time = self.samples_info_sets.iloc[val_idx].max()

            train_mask = np.ones(n_samples, dtype=bool)

            # Remove validation samples
            train_mask[val_idx] = False

            # Purge overlapping samples
            for i in indices:
                train_start = self.samples_info_sets.index[i]
                train_end = self.samples_info_sets.iloc[i]

                overlap = (
                    (train_start <= val_end_time)
                    and
                    (train_end >= val_start_time)
                )

                if overlap:
                    train_mask[i] = False

            # Embargo
            embargo_size = int(n_samples * self.pct_embargo)

            embargo_start = stop
            embargo_end = min(n_samples, stop + embargo_size)

            train_mask[embargo_start:embargo_end] = False

            train_idx = indices[train_mask]

            yield train_idx, val_idx

            current = stop