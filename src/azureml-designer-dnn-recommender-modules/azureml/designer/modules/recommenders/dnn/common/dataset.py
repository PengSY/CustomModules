from enum import Enum
import numpy as np
import pandas as pd
import tensorflow as tf
from azureml.studio.core.data_frame_schema import DataFrameSchema
from azureml.designer.modules.recommenders.dnn.common.constants import INTERACTIONS_USER_COL, INTERACTIONS_ITEM_COL, \
    FEATURES_ID_COL, INTERACTIONS_RATING_COL, RANDOM_SEED, USER_INTERNAL_KEY, ITEM_INTERNAL_KEY, RATING_INTERNAL_KEY, \
    FEATURE_INTERNAL_SUFFIX


class DatasetType(Enum):
    SingleDataset = 1
    DoubleDataset = 2
    TripleDataset = 3


class Dataset:
    def __init__(self, df: pd.DataFrame, name: str = None):
        self.df = df
        self.name = name
        self.column_attributes = self.build_column_attributes()

    def get_column_type(self, col_key):
        return self.column_attributes[col_key].column_type

    def get_element_type(self, col_key):
        return self.column_attributes[col_key].element_type

    def build_column_attributes(self):
        self.column_attributes = DataFrameSchema.generate_column_attributes(df=self.df)
        return self.column_attributes

    @property
    def column_size(self):
        return self.df.shape[1]

    @property
    def row_size(self):
        return self.df.shape[0]

    @property
    def columns(self):
        return self.df.columns


class InteractionDataset(Dataset):
    @property
    def users(self):
        return self.df.iloc[:, INTERACTIONS_USER_COL]

    @property
    def items(self):
        return self.df.iloc[:, INTERACTIONS_ITEM_COL]

    @property
    def ratings(self):
        return self.df.iloc[:, INTERACTIONS_RATING_COL]

    def preprocess(self):
        if self.column_size > 2:
            self.ratings.replace(to_replace=[np.inf, -np.inf], value=np.nan, inplace=True)
        self.df = self.df.dropna().reset_index(drop=True)

        id_cols = self.df.columns[:INTERACTIONS_RATING_COL]
        self.df[id_cols] = self.df[id_cols].astype(str)
        self.build_column_attributes()

        return self


class FeatureDataset(Dataset):
    @property
    def ids(self):
        return self.df.iloc[:, FEATURES_ID_COL]

    @property
    def features(self):
        return self.df.iloc[:, FEATURES_ID_COL + 1:]

    def preprocess_ids(self):
        # use column slice to avoid error when dataset is invalid
        self.df = self.df.dropna(subset=self.df.columns[FEATURES_ID_COL:FEATURES_ID_COL + 1]).reset_index(drop=True)
        id_col = self.df.columns[FEATURES_ID_COL:FEATURES_ID_COL + 1]
        self.df[id_col] = self.df[id_col].astype(str)

        self.build_column_attributes()
        return self


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
