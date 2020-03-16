import os
import numpy as np
import pandas as pd
import pytest
import azureml.studio.internal.error as error_setting
from azureml.designer.modules.recommenders.dnn.common.dataset import TransactionDataset, FeatureDataset
from azureml.designer.modules.recommenders.dnn.wide_and_deep.train.train_wide_and_deep_recommender import \
    TrainWideAndDeepRecommenderModule


@pytest.fixture
def entry_params():
    params_dict = dict(
        model_dir=None,
        epochs=2,
        batch_size=32,
        wide_optimizer="Adagrad",
        wide_lr=0.005,
        crossed_dim=1000,
        deep_optimizer="Adadelta",
        deep_lr=0.01,
        user_dim=8,
        item_dim=8,
        embed_dim=8,
        hidden_units="64,128,512",
        activation_fn="ReLU",
        dropout=0.8,
        batch_norm="True"
    )
    return params_dict


def _gen_null_transactions():
    transactions = None
    user_features = None
    item_features = None
    exp_error = error_setting.NullOrEmptyError
    exp_msg = 'Input "Training dataset of user-item-rating triples" is null or empty.'
    return transactions, user_features, item_features, exp_error, exp_msg


def _gen_transactions_invalid_column_number():
    transactions = {'user': [1, 2, 3], 'item': [4, 5, 6]}
    user_features = None
    item_features = None
    exp_error = error_setting.UnexpectedNumberOfColumnsError
    exp_msg = r'In input dataset "Training dataset of user-item-rating triples", ' \
              r'expected "3" column\(s\) but found "2" column\(s\) instead.'
    return transactions, user_features, item_features, exp_error, exp_msg


def _gen_empty_transactions():
    transactions = {'user': [], 'item': [], 'rating': []}
    user_features = None
    item_features = None
    exp_error = error_setting.TooFewRowsInDatasetError
    exp_msg = r'Number of rows in input dataset "Training dataset of user-item-rating triples" is 0, ' \
              r'less than allowed minimum of 1 row\(s\).'
    return transactions, user_features, item_features, exp_error, exp_msg


def _gen_transactions_with_invalid_rating_column():
    transactions = {'user': [1, 2, 3], 'item': [4, 5, 6], 'rating': ['a', 'b', 'c']}
    user_features = None
    item_features = None
    exp_error = error_setting.InvalidColumnTypeError
    exp_msg = 'Cannot process column "rating" of type String. The type is not supported by the module.'
    return transactions, user_features, item_features, exp_error, exp_msg


def _gen_transactions_with_duplicated_ratings():
    transactions = {'user': [1, 2, 2], 'item': [3, 4, 4], 'rating': [1, 2, 3]}
    user_features = None
    item_features = None
    exp_error = error_setting.MoreThanOneRatingError
    exp_msg = r'More than one rating exist for the value\(s\) in dataset.'
    return transactions, user_features, item_features, exp_error, exp_msg


def _gen_transactions_with_all_ratings_missing():
    transactions = {'user': [1, 2, 2], 'item': [3, 4, 4], 'rating': [np.nan, np.nan, np.nan]}
    user_features = None
    item_features = None
    exp_error = error_setting.InvalidColumnTypeError
    exp_msg = 'Cannot process column "rating" of type NAN. The type is not supported by the module.'
    return transactions, user_features, item_features, exp_error, exp_msg


def _gen_transactions_with_none_valid_samples():
    transactions = {'user': [np.nan, 2, 3], 'item': [1, np.nan, 3], 'rating': [1, 2, np.nan]}
    user_features = None
    item_features = None
    exp_error = error_setting.InvalidDatasetError
    exp_msg = 'Training dataset of user-item-rating triples contains invalid data, ' \
              'dataset does not have any valid samples.'
    return transactions, user_features, item_features, exp_error, exp_msg


def _gen_user_features_with_insufficient_columns():
    transactions = {'user': [1, 2, 3], 'item': [4, 5, 6], 'rating': [7, 8, 9]}
    user_features = {'user': [1, 2, 3]}
    item_features = None
    exp_error = error_setting.TooFewColumnsInDatasetError
    exp_msg = r'Number of columns in input dataset "User features" is less than allowed minimum of 2 column\(s\).'
    return transactions, user_features, item_features, exp_error, exp_msg


def _gen_empty_user_features():
    transactions = {'user': [1, 2, 3], 'item': [4, 5, 6], 'rating': [7, 8, 9]}
    user_features = {'user': [], 'age': []}
    item_features = None
    exp_error = error_setting.TooFewRowsInDatasetError
    exp_msg = r'Number of rows in input dataset "User features" is 0, less than allowed minimum of 1 row\(s\).'
    return transactions, user_features, item_features, exp_error, exp_msg


def _gen_user_features_with_invalid_feature_type():
    transactions = {'user': [1, 2, 3], 'item': [4, 5, 6], 'rating': [7, 8, 9]}
    user_features = {'user': [1, 2], 'interest': [np.nan, np.nan]}
    item_features = None
    exp_error = error_setting.InvalidColumnTypeError
    exp_msg = 'Cannot process column "interest" of type NAN. The type is not supported by the module.'
    return transactions, user_features, item_features, exp_error, exp_msg


def _gen_user_features_with_duplicated_ids():
    transactions = {'user': [1, 2, 3], 'item': [4, 5, 6], 'rating': [7, 8, 9]}
    user_features = {'user': [1, 2, 2], 'age': [30, 31, 32]}
    item_features = None
    exp_error = error_setting.DuplicateFeatureDefinitionError
    exp_msg = "Duplicate feature definition for a user or item."
    return transactions, user_features, item_features, exp_error, exp_msg


def _gen_user_features_with_none_valid_ids():
    transactions = {'user': [1, 2, 3], 'item': [4, 5, 6], 'rating': [7, 8, 9]}
    user_features = {'user': [np.nan, None], 'age': [1, 2]}
    item_features = None
    exp_error = error_setting.InvalidColumnTypeError
    exp_msg = 'Cannot process column "user" of type NAN. The type is not supported by the module.'
    return transactions, user_features, item_features, exp_error, exp_msg


def _gen_item_features_with_insufficient_columns():
    transactions = {'user': [1, 2, 3], 'item': [4, 5, 6], 'rating': [7, 8, 9]}
    user_features = None
    item_features = {'item': [1, 2, 3]}
    exp_error = error_setting.TooFewColumnsInDatasetError
    exp_msg = r'Number of columns in input dataset "Item features" is less than allowed minimum of 2 column\(s\).'
    return transactions, user_features, item_features, exp_error, exp_msg


def _gen_empty_item_features():
    transactions = {'user': [1, 2, 3], 'item': [4, 5, 6], 'rating': [7, 8, 9]}
    user_features = None
    item_features = {'item': [], 'price': []}
    exp_error = error_setting.TooFewRowsInDatasetError
    exp_msg = r'Number of rows in input dataset "Item features" is 0, less than allowed minimum of 1 row\(s\).'
    return transactions, user_features, item_features, exp_error, exp_msg


def _gen_item_features_with_invalid_feature_type():
    transactions = {'user': [1, 2, 3], 'item': [4, 5, 6], 'rating': [7, 8, 9]}
    user_features = None
    item_features = {'item': [1, 2], 'category': [np.nan, np.nan]}
    exp_error = error_setting.InvalidColumnTypeError
    exp_msg = 'Cannot process column "category" of type NAN. The type is not supported by the module.'
    return transactions, user_features, item_features, exp_error, exp_msg


def _gen_item_features_with_duplicated_ids():
    transactions = {'user': [1, 2, 3], 'item': [4, 5, 6], 'rating': [7, 8, 9]}
    user_features = None
    item_features = {'item': [2, 2], 'price': [3, 4]}
    exp_error = error_setting.DuplicateFeatureDefinitionError
    exp_msg = "Duplicate feature definition for a user or item."
    return transactions, user_features, item_features, exp_error, exp_msg


def _gen_item_features_with_none_valid_ids():
    transactions = {'user': [1, 2, 3], 'item': [4, 5, 6], 'rating': [7, 8, 9]}
    user_features = None
    item_features = {'item': [np.nan, None], 'price': [1, 2]}
    exp_error = error_setting.InvalidColumnTypeError
    exp_msg = 'Cannot process column "item" of type NAN. The type is not supported by the module.'
    return transactions, user_features, item_features, exp_error, exp_msg


@pytest.mark.parametrize('transactions,user_features,item_features,exp_error,exp_msg', [
    _gen_null_transactions(),
    _gen_transactions_invalid_column_number(),
    _gen_empty_transactions(),
    _gen_transactions_with_invalid_rating_column(),
    _gen_transactions_with_duplicated_ratings(),
    _gen_transactions_with_all_ratings_missing(),
    _gen_transactions_with_none_valid_samples(),
    _gen_user_features_with_insufficient_columns(),
    _gen_empty_user_features(),
    _gen_user_features_with_invalid_feature_type(),
    _gen_user_features_with_duplicated_ids(),
    _gen_user_features_with_none_valid_ids(),
    _gen_item_features_with_insufficient_columns(),
    _gen_empty_item_features(),
    _gen_item_features_with_invalid_feature_type(),
    _gen_item_features_with_duplicated_ids(),
    _gen_item_features_with_none_valid_ids(),
])
def test_error_case(transactions, user_features, item_features, exp_error, exp_msg, entry_params):
    with pytest.raises(expected_exception=exp_error, match=exp_msg):
        if transactions is not None:
            transactions = TransactionDataset(pd.DataFrame(transactions))
        if user_features is not None:
            user_features = FeatureDataset(pd.DataFrame(user_features))
        if item_features is not None:
            item_features = FeatureDataset(pd.DataFrame(item_features))
        TrainWideAndDeepRecommenderModule().run(transactions=transactions, user_features=user_features,
                                                item_features=item_features, **entry_params)


def _gen_datasets_without_features():
    transactions = {'user': [1, 2, 3], 'item': [4, 5, 6], 'rating': [7, 8, 8]}
    user_features = None
    item_features = None
    return transactions, user_features, item_features


def _gen_datasets_with_missing_part_values():
    transactions = {'user': [1, 2, np.nan, 10], 'item': [3, 3, 4, 8], 'rating': [np.nan, 5, 6, 9]}
    user_features = {'user': [np.nan, 2], 'age': [33, 32]}
    item_features = {'item': [1, 3], 'price': [np.nan, 50]}
    return transactions, user_features, item_features


def _gen_datasets_with_multi_type_ids():
    dataset = {'user': [True, 1, 'bob', 1.84], 'item': ['The Avengers', False, 6.8, 2], 'rating': [1, 2, 3, 4]}
    user_features = {'user': [True, 1, 'bob', 1.84], 'age': [1, 2, 3, 4]}
    item_features = {'item': ['The Avengers', False, 6.8, 2], 'price': ['high', 'low', 'high', 'low']}
    return dataset, user_features, item_features


def _gen_user_features_with_object_feature_type():
    transactions = {'user': [1, 2, 3], 'item': [4, 5, 6], 'rating': [7, 8, 9]}
    user_features = {'user': [1, 2], 'interest': [[1, 2], [3, 4]]}
    item_features = None
    return transactions, user_features, item_features


def _gen_user_features_with_none_valid_features():
    transactions = {'user': [1, 2, 3], 'item': [4, 5, 6], 'rating': [7, 8, 9]}
    user_features = {'user': [5, 6], 'age': [1, 2]}
    item_features = None
    return transactions, user_features, item_features


def _gen_item_features_with_object_feature_type():
    transactions = {'user': [1, 2, 3], 'item': [4, 5, 6], 'rating': [7, 8, 9]}
    user_features = None
    item_features = {'item': [1, 2], 'category': [[1, 2], [3, 4]]}
    return transactions, user_features, item_features


def _gen_item_features_with_none_valid_features():
    transactions = {'user': [1, 2, 3], 'item': [4, 5, 6], 'rating': [7, 8, 9]}
    user_features = None
    item_features = {'item': [1, 2], 'price': ['high', 'low']}
    return transactions, user_features, item_features


@pytest.mark.parametrize('transactions,user_features,item_features', [
    _gen_datasets_without_features(),
    _gen_datasets_with_missing_part_values(),
    _gen_datasets_with_multi_type_ids(),
    _gen_user_features_with_object_feature_type(),
    _gen_user_features_with_none_valid_features(),
    _gen_item_features_with_object_feature_type(),
    _gen_item_features_with_none_valid_features(),
])
def test_valid_corner_case(transactions, user_features, item_features, tmp_path, entry_params):
    if transactions is not None:
        transactions = TransactionDataset(pd.DataFrame(transactions))
    if user_features is not None:
        user_features = FeatureDataset(pd.DataFrame(user_features))
    if item_features is not None:
        item_features = FeatureDataset(pd.DataFrame(item_features))
    entry_params['model_dir'] = tmp_path
    TrainWideAndDeepRecommenderModule().run(transactions=transactions, user_features=user_features,
                                            item_features=item_features, **entry_params)


def test_sample_data(entry_params, tmp_path):
    input_dir = os.path.join(os.path.dirname(__file__), 'inputs')
    transactions_file = os.path.join(input_dir, 'restaurant_ratings_train.csv')
    user_features_file = os.path.join(input_dir, 'restaurant_user_features.csv')
    item_features_file = os.path.join(input_dir, 'restaurant_item_features.csv')
    transactions = TransactionDataset(pd.read_csv(transactions_file))
    user_features = FeatureDataset(pd.read_csv(user_features_file))
    item_features = FeatureDataset(pd.read_csv(item_features_file))
    entry_params['model_dir'] = tmp_path
    entry_params['epochs'] = 10
    TrainWideAndDeepRecommenderModule().run(transactions=transactions, user_features=user_features,
                                            item_features=item_features, **entry_params)
