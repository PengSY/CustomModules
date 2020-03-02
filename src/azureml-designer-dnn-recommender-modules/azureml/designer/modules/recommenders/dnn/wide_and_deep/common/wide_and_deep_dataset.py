import pandas as pd
import tensorflow as tf
from azureml.designer.modules.recommenders.dnn.common.constants import RANDOM_SEED, USER_INTERNAL_KEY, \
    ITEM_INTERNAL_KEY, RATING_INTERNAL_KEY, FEATURE_INTERNAL_SUFFIX
from azureml.designer.modules.recommenders.dnn.common.dataset import Dataset


class WideDeepDataset(Dataset):
    def __init__(self, interactions: Dataset, user_features: Dataset, item_features: Dataset, name: str = None):
        df = WideDeepDataset._generate(interactions, user_features=user_features, item_features=item_features)
        self._interactions = df.iloc[:, :interactions.column_size]
        self._features = df.iloc[:, interactions.column_size:]
        super().__init__(df=df, name=name)

    @staticmethod
    def _rename_interactions(interactions: pd.DataFrame):
        internal_keys = [USER_INTERNAL_KEY, ITEM_INTERNAL_KEY, RATING_INTERNAL_KEY]
        interactions = interactions.rename(columns=dict(zip(interactions.columns, internal_keys)))
        return interactions

    @staticmethod
    def _rename_features(features: pd.DataFrame, internal_id_key: str):
        internal_keys = [internal_id_key] + [f'{internal_id_key}{FEATURE_INTERNAL_SUFFIX}_{i}' for i in
                                             range(1, features.shape[1])]

        features = features.rename(columns=dict(zip(features.columns, internal_keys)))
        return features

    @staticmethod
    def _generate(interactions: Dataset, user_features: Dataset = None, item_features: Dataset = None):
        interactions = WideDeepDataset._rename_interactions(interactions.df)
        df = interactions
        if user_features is not None:
            user_features = WideDeepDataset._rename_features(user_features.df, USER_INTERNAL_KEY)
            df = WideDeepDataset._merge_features(df=df, features=user_features, key=USER_INTERNAL_KEY)
        if item_features is not None:
            item_features = WideDeepDataset._rename_features(item_features.df, ITEM_INTERNAL_KEY)
            df = WideDeepDataset._merge_features(df=df, features=item_features, key=ITEM_INTERNAL_KEY)

        return df

    @staticmethod
    def _merge_features(df: pd.DataFrame, features: pd.DataFrame, key: str):
        df = pd.merge(left=df, right=features, on=key, how='left')
        return df

    def get_input_handler(self, batch_size, epochs=1, shuffle=False):
        X = self.df.to_dict("list")
        if RATING_INTERNAL_KEY in self.df:
            y = X.pop(self.ratings.name)
        else:
            y = None

        return lambda: WideDeepDataset._dataset(X, y, batch_size, epochs, shuffle)

    @staticmethod
    def _dataset(X, y, batch_size, epochs, shuffle=False):
        if y is None:
            dataset = tf.data.Dataset.from_tensor_slices(X)
        else:
            dataset = tf.data.Dataset.from_tensor_slices((X, y))
        if shuffle:
            dataset = dataset.shuffle(len(y), seed=RANDOM_SEED, reshuffle_each_iteration=True)

        return dataset.repeat(epochs).batch(batch_size)

    @property
    def users(self):
        return self._interactions[USER_INTERNAL_KEY]

    @property
    def items(self):
        return self._interactions[ITEM_INTERNAL_KEY]

    @property
    def ratings(self):
        return self._interactions[RATING_INTERNAL_KEY]

    @property
    def features(self):
        return self._features
