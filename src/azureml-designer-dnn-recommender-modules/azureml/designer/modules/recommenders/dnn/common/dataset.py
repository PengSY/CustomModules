import pandas as pd
from azureml.studio.core.data_frame_schema import DataFrameSchema
from azureml.designer.modules.recommenders.dnn.common.constants import TRANSACTIONS_USER_COL, TRANSACTIONS_ITEM_COL, \
    FEATURES_ID_COL, TRANSACTIONS_RATING_COL
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


class TransactionDataset(Dataset):
    @property
    def users(self):
        return self.df.iloc[:, TRANSACTIONS_USER_COL]

    @users.setter
    def users(self, users: pd.Series):
        self.df.iloc[:, TRANSACTIONS_USER_COL] = users

    @property
    def items(self):
        return self.df.iloc[:, TRANSACTIONS_ITEM_COL]

    @items.setter
    def items(self, items: pd.Series):
        self.df.iloc[:, TRANSACTIONS_ITEM_COL] = items

    @property
    def ratings(self):
        if self.column_size - 1 >= TRANSACTIONS_RATING_COL:
            return self.df.iloc[:, TRANSACTIONS_RATING_COL]
        else:
            return None

    @ratings.setter
    def ratings(self, ratings):
        if self.column_size - 1 >= TRANSACTIONS_RATING_COL:
            self.df.iloc[:, TRANSACTIONS_RATING_COL] = ratings
        else:
            self.df[ratings.name] = ratings


class FeatureDataset(Dataset):
    @property
    def ids(self):
        return self.df.iloc[:, FEATURES_ID_COL]

    @ids.setter
    def ids(self, ids):
        self.df.iloc[:, FEATURES_ID_COL] = ids

    @property
    def features(self):
        return self.df.iloc[:, FEATURES_ID_COL + 1:]
