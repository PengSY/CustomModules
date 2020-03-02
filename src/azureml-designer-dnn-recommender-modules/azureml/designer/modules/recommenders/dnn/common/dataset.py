import numpy as np
import pandas as pd
from azureml.studio.core.data_frame_schema import DataFrameSchema
from azureml.designer.modules.recommenders.dnn.common.constants import INTERACTIONS_USER_COL, INTERACTIONS_ITEM_COL, \
    FEATURES_ID_COL, INTERACTIONS_RATING_COL
from azureml.designer.modules.recommenders.dnn.common.entry_param import EntryParam
from azureml.studio.core.io.data_frame_directory import load_data_frame_from_directory


class Dataset(metaclass=EntryParam):
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

    @classmethod
    def load(cls, path: str):
        df = load_data_frame_from_directory(load_from_dir=path).data
        return cls(df=df)


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
