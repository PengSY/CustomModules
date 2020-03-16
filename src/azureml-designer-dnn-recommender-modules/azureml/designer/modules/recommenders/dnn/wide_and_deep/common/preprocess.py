import numpy as np
from azureml.designer.modules.recommenders.dnn.common.dataset import TransactionDataset, FeatureDataset
from azureml.designer.modules.recommenders.dnn.common.utils import convert_to_str


def preprocess_transactions(transactions: TransactionDataset):
    if transactions.ratings is not None:
        transactions.ratings = transactions.ratings.replace(to_replace=[np.inf, -np.inf], value=np.nan)
    transactions.users = convert_to_str(transactions.users)
    transactions.items = convert_to_str(transactions.items)
    transactions.df = transactions.df.dropna().reset_index(drop=True)
    transactions.build_column_attributes()

    return transactions


def preprocess_features(features: FeatureDataset):
    features.df = features.df.dropna(subset=[features.ids.name]).reset_index(drop=True)
    features.ids = convert_to_str(features.ids)
    features.build_column_attributes()

    return features
