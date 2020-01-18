import math
import pytest
import os
import numpy as np
import pandas as pd
import azureml.studio.internal.error as error_setting
from azureml.designer.modules.recommenders.dnn.common.dataset import FeatureDataset, InteractionDataset
from azureml.designer.modules.recommenders.dnn.wide_and_deep.score. \
    score_wide_and_deep_recommender import ScoreWideAndDeepRecommenderModule
from azureml.designer.modules.recommenders.dnn.common.extend_types import RecommendedItemSelection, \
    RecommenderPredictionKind
from azureml.designer.modules.recommenders.dnn.common.constants import INTERACTIONS_USER_COL, INTERACTIONS_ITEM_COL, \
    USER_INTERNAL_KEY, ITEM_INTERNAL_KEY


def get_input_dir():
    return os.path.join(os.path.dirname(__file__), 'inputs')


def default_params():
    normal_params = dict(
        prediction_kind=RecommenderPredictionKind.ItemRecommendation,
        recommended_item_selection=RecommendedItemSelection.FromRatedItems,
        max_recommended_item_count=3,
        min_recommendation_pool_size=2,
        return_ratings="False",
    )
    port_params = dict(
        learner=os.path.join(get_input_dir(), 'model_dir'),
        test_interactions=InteractionDataset(pd.read_csv(os.path.join(get_input_dir(), 'restaurant_ratings_test.csv'))),
        user_features=None,
        item_features=None,
        training_interactions=InteractionDataset(
            pd.read_csv(os.path.join(get_input_dir(), 'restaurant_ratings_train.csv'))),
    )
    return normal_params, port_params


def _gen_null_learner():
    normal_params, port_params = default_params()
    port_params['learner'] = None
    exp_error = error_setting.NullOrEmptyError
    exp_msg = 'Input "Trained Wide and Deep recommendation model" is null or empty.'
    return normal_params, port_params, exp_error, exp_msg


def _gen_invalid_learner():
    normal_params, port_params = default_params()
    port_params['learner'] = os.path.join(get_input_dir(), 'invalid_model_dir')
    port_params['test_interactions'] = InteractionDataset(pd.DataFrame({'userID': ['U1138'], 'placeID': [132925]}))
    exp_error = error_setting.InvalidLearnerError
    exp_msg = 'Learner "Trained Wide and Deep recommendation model" has invalid type.'
    return normal_params, port_params, exp_error, exp_msg


def _gen_null_interactions():
    normal_params, port_params = default_params()
    port_params['test_interactions'] = None
    exp_error = error_setting.NullOrEmptyError
    exp_msg = 'Input "Dataset to score" is null or empty.'
    return normal_params, port_params, exp_error, exp_msg


def _gen_empty_interactions():
    normal_params, port_params = default_params()
    port_params['test_interactions'] = InteractionDataset(pd.DataFrame({'user': [], 'item': []}))
    exp_error = error_setting.TooFewRowsInDatasetError
    exp_msg = r'Number of rows in input dataset "Dataset to score" is 0, less than allowed minimum of 1 row\(s\).'
    return normal_params, port_params, exp_error, exp_msg


def _gen_interactions_with_invalid_column_number(prediction_kind=RecommenderPredictionKind.ItemRecommendation,
                                                 recommendation_item_selection=RecommendedItemSelection.FromRatedItems):
    normal_params, port_params = default_params()
    normal_params['prediction_kind'] = prediction_kind
    normal_params['recommended_item_selection'] = recommendation_item_selection
    if prediction_kind == RecommenderPredictionKind.RatingPrediction:
        port_params['test_interactions'] = InteractionDataset(pd.DataFrame({'user': [1, 2, 3]}))
        exp_error = error_setting.TooFewColumnsInDatasetError
        exp_msg = r'Number of columns in input dataset "Dataset to score" is less ' \
                  r'than allowed minimum of 2 column\(s\).'
    else:
        if recommendation_item_selection == RecommendedItemSelection.FromRatedItems:
            port_params['test_interactions'] = InteractionDataset(
                pd.DataFrame({'user': [1, 2]}))
            exp_error = error_setting.TooFewColumnsInDatasetError
            exp_msg = r'Number of columns in input dataset "Dataset to score" is less ' \
                      r'than allowed minimum of 2 column\(s\).'
        elif recommendation_item_selection == RecommendedItemSelection.FromAllItems:
            port_params['test_interactions'] = InteractionDataset(
                pd.DataFrame({'user': [1], 'item': [2], 'rating': [3], 'extra': [4]}))
            exp_error = error_setting.UnexpectedNumberOfColumnsError
            exp_msg = 'Unexpected number of columns in the dataset "Dataset to score".'
        else:
            port_params['test_interactions'] = InteractionDataset(pd.DataFrame({'user': [1, 2]}))
            exp_error = error_setting.TooFewColumnsInDatasetError
            exp_msg = r'Number of columns in input dataset "Dataset to score" is less ' \
                      r'than allowed minimum of 2 column\(s\).'
    return normal_params, port_params, exp_error, exp_msg


def _gen_empty_user_features():
    normal_params, port_params = default_params()
    port_params['user_features'] = FeatureDataset(pd.DataFrame({'user': [], 'age': []}))
    exp_error = error_setting.TooFewRowsInDatasetError
    exp_msg = r'Number of rows in input dataset "User features" is 0, less than allowed minimum of 1 row\(s\).'
    return normal_params, port_params, exp_error, exp_msg


def _gen_user_features_with_invalid_column_number():
    normal_params, port_params = default_params()
    port_params['user_features'] = FeatureDataset(pd.DataFrame({'user': []}))
    exp_error = error_setting.TooFewColumnsInDatasetError
    exp_msg = r'Number of columns in input dataset "User features" is less than allowed minimum of 2 column\(s\).'
    return normal_params, port_params, exp_error, exp_msg


def _gen_user_features_with_invalid_feature_type():
    normal_params, port_params = default_params()
    port_params['user_features'] = FeatureDataset(pd.DataFrame({'user': [1], 'interest': [[1, 2]]}))
    exp_error = error_setting.InvalidColumnTypeError
    exp_msg = 'Cannot process column "interest" of type Object. The type is not supported by the module.'
    return normal_params, port_params, exp_error, exp_msg


def _gen_user_features_with_duplicated_ids():
    normal_params, port_params = default_params()
    port_params['user_features'] = FeatureDataset(pd.DataFrame({'user': [1, 2, 2], 'age': [30, 31, 32]}))
    exp_error = error_setting.DuplicateFeatureDefinitionError
    exp_msg = "Duplicate feature definition for a user or item."
    return normal_params, port_params, exp_error, exp_msg


def _gen_user_features_with_none_valid_ids():
    normal_params, port_params = default_params()
    normal_params['prediction_kind'] = RecommenderPredictionKind.RatingPrediction
    port_params['test_interactions'] = InteractionDataset(pd.DataFrame({'user': [1], 'item': [2]}))
    port_params['user_features'] = FeatureDataset(pd.DataFrame({'user': [np.nan, None],
                                                                'latitude': [1, 2],
                                                                'longitude': [1, 2],
                                                                'interest': ['a', 'b'],
                                                                'personality': ['a', 'b']}))
    exp_error = error_setting.InvalidColumnTypeError
    exp_msg = 'Cannot process column "user" of type NAN. The type is not supported by the module.'
    return normal_params, port_params, exp_error, exp_msg


def _gen_user_features_with_incompatible_feature_types():
    normal_params, port_params = default_params()
    port_params['user_features'] = FeatureDataset(pd.DataFrame({'user': [1, 2],
                                                                'latitude': [1, 2],
                                                                'longitude': ['1', '2'],
                                                                'interest': ['a', 'b'],
                                                                'personality': ['a', 'b']}))
    exp_error = error_setting.InvalidColumnTypeError
    exp_msg = 'Cannot process column "longitude" of type String. The type is not supported by the module.'
    return normal_params, port_params, exp_error, exp_msg


def _gen_user_features_with_missing_feature_column():
    normal_params, port_params = default_params()
    port_params['user_features'] = FeatureDataset(pd.DataFrame({'user': [1, 2],
                                                                'latitude': [1, 2],
                                                                'longitude': [1, 2],
                                                                'personality': ['a', 'b']}))
    exp_error = error_setting.ColumnNotFoundError
    exp_msg = 'Column with name or index "interest" does not exist in "User features".'
    return normal_params, port_params, exp_error, exp_msg


def _gen_empty_item_features():
    normal_params, port_params = default_params()
    port_params['item_features'] = FeatureDataset(pd.DataFrame({'item': [], 'price': []}))
    exp_error = error_setting.TooFewRowsInDatasetError
    exp_msg = r'Number of rows in input dataset "Item features" is 0, less than allowed minimum of 1 row\(s\).'
    return normal_params, port_params, exp_error, exp_msg


def _gen_item_features_with_invalid_column_number():
    normal_params, port_params = default_params()
    port_params['item_features'] = FeatureDataset(pd.DataFrame({'item': []}))
    exp_error = error_setting.TooFewColumnsInDatasetError
    exp_msg = r'Number of columns in input dataset "Item features" is less than allowed minimum of 2 column\(s\).'
    return normal_params, port_params, exp_error, exp_msg


def _gen_item_features_with_invalid_feature_type():
    normal_params, port_params = default_params()
    port_params['item_features'] = FeatureDataset(pd.DataFrame({'item': [1], 'price': [[1, 2]]}))
    exp_error = error_setting.InvalidColumnTypeError
    exp_msg = 'Cannot process column "price" of type Object. The type is not supported by the module.'
    return normal_params, port_params, exp_error, exp_msg


def _gen_item_features_with_duplicated_ids():
    normal_params, port_params = default_params()
    port_params['item_features'] = FeatureDataset(pd.DataFrame({'item': [1, 2, 2], 'price': [30, 31, 32]}))
    exp_error = error_setting.DuplicateFeatureDefinitionError
    exp_msg = "Duplicate feature definition for a user or item."
    return normal_params, port_params, exp_error, exp_msg


def _gen_item_features_with_none_valid_ids():
    normal_params, port_params = default_params()
    normal_params['prediction_kind'] = RecommenderPredictionKind.RatingPrediction
    port_params['test_interactions'] = InteractionDataset(pd.DataFrame({'user': [1], 'item': [2]}))
    port_params['item_features'] = FeatureDataset(pd.DataFrame({'item': [np.nan, None],
                                                                'latitude': [1, 2],
                                                                'longitude': [1, 2],
                                                                'price': ['low', 'high']}))
    exp_error = error_setting.InvalidColumnTypeError
    exp_msg = 'Cannot process column "item" of type NAN. The type is not supported by the module.'
    return normal_params, port_params, exp_error, exp_msg


def _gen_item_features_with_incompatible_feature_types():
    normal_params, port_params = default_params()
    port_params['item_features'] = FeatureDataset(pd.DataFrame({'item': [1, 2],
                                                                'latitude': [1, 2],
                                                                'longitude': [1, 2],
                                                                'price': [5, 6]}))
    exp_error = error_setting.InvalidColumnTypeError
    exp_msg = 'Cannot process column "price" of type Numeric. The type is not supported by the module.'
    return normal_params, port_params, exp_error, exp_msg


def _gen_item_features_with_missing_feature_column():
    normal_params, port_params = default_params()
    port_params['item_features'] = FeatureDataset(pd.DataFrame({'user': [1, 2],
                                                                'latitude': [1, 2],
                                                                'longitude': [1, 2]}))
    exp_error = error_setting.ColumnNotFoundError
    exp_msg = 'Column with name or index "price" does not exist in "Item features".'
    return normal_params, port_params, exp_error, exp_msg


def _gen_null_training_data():
    normal_params, port_params = default_params()
    normal_params['recommended_item_selection'] = RecommendedItemSelection.FromUnratedItems
    port_params['training_interactions'] = None
    exp_error = error_setting.NullOrEmptyError
    exp_msg = 'Input "Training data" is null or empty.'
    return normal_params, port_params, exp_error, exp_msg


def _gen_empty_training_data():
    normal_params, port_params = default_params()
    normal_params['recommended_item_selection'] = RecommendedItemSelection.FromUnratedItems
    port_params['training_interactions'] = InteractionDataset(pd.DataFrame({'user': [], 'item': []}))
    exp_error = error_setting.TooFewRowsInDatasetError
    exp_msg = r'Number of rows in input dataset "Training data" is 0, less than allowed minimum of 1 row\(s\).'
    return normal_params, port_params, exp_error, exp_msg


def _gen_training_data_with_insufficient_columns():
    normal_params, port_params = default_params()
    normal_params['recommended_item_selection'] = RecommendedItemSelection.FromUnratedItems
    port_params['training_interactions'] = InteractionDataset(pd.DataFrame({'user': [1, 2]}))
    exp_error = error_setting.TooFewColumnsInDatasetError
    exp_msg = r'Number of columns in input dataset "Training data" is less than allowed minimum of 2 column\(s\).'
    return normal_params, port_params, exp_error, exp_msg


def _gen_training_data_with_invalid_column_number():
    normal_params, port_params = default_params()
    normal_params['recommended_item_selection'] = RecommendedItemSelection.FromUnratedItems
    port_params['training_interactions'] = InteractionDataset(
        pd.DataFrame({'user': [1, 2], 'item': [34, 5], 'rating': [7, 8], 'extra': [1, 2]}))
    exp_error = error_setting.UnexpectedNumberOfColumnsError
    exp_msg = 'Unexpected number of columns in the dataset "Training data".'
    return normal_params, port_params, exp_error, exp_msg


@pytest.mark.parametrize(
    'normal_params,port_params, exp_error, exp_msg', [
        _gen_null_learner(),
        _gen_invalid_learner(),
        _gen_null_interactions(),
        _gen_empty_interactions(),
        _gen_interactions_with_invalid_column_number(RecommenderPredictionKind.RatingPrediction),
        _gen_interactions_with_invalid_column_number(),
        _gen_interactions_with_invalid_column_number(
            recommendation_item_selection=RecommendedItemSelection.FromAllItems),
        _gen_interactions_with_invalid_column_number(
            recommendation_item_selection=RecommendedItemSelection.FromUnratedItems),
        _gen_empty_user_features(),
        _gen_user_features_with_invalid_column_number(),
        _gen_user_features_with_invalid_feature_type(),
        _gen_user_features_with_duplicated_ids(),
        _gen_user_features_with_none_valid_ids(),
        _gen_user_features_with_incompatible_feature_types(),
        _gen_user_features_with_missing_feature_column(),
        _gen_empty_item_features(),
        _gen_item_features_with_invalid_column_number(),
        _gen_item_features_with_invalid_feature_type(),
        _gen_item_features_with_duplicated_ids(),
        _gen_item_features_with_none_valid_ids(),
        _gen_item_features_with_incompatible_feature_types(),
        _gen_item_features_with_missing_feature_column(),
        _gen_null_training_data(),
        _gen_empty_training_data(),
        _gen_training_data_with_insufficient_columns(),
        _gen_training_data_with_invalid_column_number(),
    ])
def test_error_case(normal_params, port_params, exp_error, exp_msg):
    with pytest.raises(expected_exception=exp_error, match=exp_msg):
        ScoreWideAndDeepRecommenderModule(**normal_params).run(**port_params)


def test_user_features_with_none_valid_features():
    normal_params, port_params = default_params()
    port_params['test_interactions'] = InteractionDataset(pd.DataFrame({'user': [1, 2], 'item': [3, 4]}))
    normal_params['prediction_kind'] = RecommenderPredictionKind.RatingPrediction
    port_params['user_features'] = FeatureDataset(pd.DataFrame({'user': [99, 999],
                                                                'latitude': [1, 2],
                                                                'longitude': [1, 2],
                                                                'interest': ['a', 'b'],
                                                                'personality': ['a', 'b']}))
    res_df = ScoreWideAndDeepRecommenderModule(**normal_params).run(**port_params)
    assert all(res_df.columns == ['User', 'Item', 'Rating'])
    assert res_df.shape[0] == 2


def test_item_features_with_none_valid_features():
    normal_params, port_params = default_params()
    port_params['test_interactions'] = InteractionDataset(pd.DataFrame({'user': [1, 2], 'item': [3, 4]}))
    normal_params['prediction_kind'] = RecommenderPredictionKind.RatingPrediction
    normal_params['item_features'] = FeatureDataset(pd.DataFrame({'item': [100, 1000],
                                                                  'latitude': [1, 2],
                                                                  'longitude': [3, 4],
                                                                  'price': ['low', 'high']}))
    res_df = ScoreWideAndDeepRecommenderModule(**normal_params).run(**port_params)
    assert all(res_df.columns == ['User', 'Item', 'Rating'])
    assert res_df.shape[0] == 2


def test_item_recommendation_with_none_valid_samples():
    normal_params, port_params = default_params()
    port_params['test_interactions'] = InteractionDataset(pd.DataFrame({'user': [1, np.nan], 'item': [np.nan, 2]}))
    res_df = ScoreWideAndDeepRecommenderModule(**normal_params).run(**port_params)
    assert all(res_df.columns == ['User', 'Item 1', 'Item 2', 'Item 3'])
    assert res_df.shape[0] == 0


def test_item_recommendation_with_duplicated_pairs():
    normal_params, port_params = default_params()
    port_params['test_interactions'] = InteractionDataset(pd.DataFrame({'user': [1, 1], 'item': [2, 2]}))
    normal_params['min_recommendation_pool_size'] = 1
    res_df = ScoreWideAndDeepRecommenderModule(**normal_params).run(**port_params)
    assert (res_df.values == [['1', '2', None, None]]).all()


def test_rating_prediction_with_none_valid_samples():
    normal_params, port_params = default_params()
    port_params['test_interactions'] = InteractionDataset(pd.DataFrame({'user': [1, np.nan], 'item': [np.nan, 2]}))
    normal_params['prediction_kind'] = RecommenderPredictionKind.RatingPrediction
    res_df = ScoreWideAndDeepRecommenderModule(**normal_params).run(**port_params)
    assert all(res_df.columns == ['User', 'Item', 'Rating'])
    assert res_df.shape[0] == 0


def test_rating_prediction_with_duplicated_pairs():
    normal_params, port_params = default_params()
    port_params['test_interactions'] = InteractionDataset(pd.DataFrame({'user': [1, 1], 'item': [2, 2]}))
    normal_params['prediction_kind'] = RecommenderPredictionKind.RatingPrediction
    res_df = ScoreWideAndDeepRecommenderModule(**normal_params).run(**port_params)
    assert (res_df.values[:, :2] == [['1', '2']]).all()


def test_rating_prediction_with_sample_data():
    normal_params, port_params = default_params()
    normal_params['prediction_kind'] = RecommenderPredictionKind.RatingPrediction
    normal_params['recommended_item_selection'] = None
    normal_params['max_recommended_item_count'] = None
    normal_params['min_recommendation_pool_size'] = None
    normal_params['return_ratings'] = None
    port_params['training_interactions'] = None
    res_df = ScoreWideAndDeepRecommenderModule(**normal_params).run(**port_params)

    assert all(res_df.columns == ['User', 'Item', 'Rating'])

    test_interactions_df = port_params['test_interactions'].df.iloc[:, [INTERACTIONS_USER_COL, INTERACTIONS_ITEM_COL]]
    test_interactions_df = test_interactions_df.rename(
        columns=dict(zip(test_interactions_df.columns, [USER_INTERNAL_KEY, ITEM_INTERNAL_KEY])))
    test_interactions_df = test_interactions_df.sort_values([USER_INTERNAL_KEY, ITEM_INTERNAL_KEY]).astype(str)
    res_df = res_df.loc[:, [USER_INTERNAL_KEY, ITEM_INTERNAL_KEY]]
    res_df = res_df.sort_values([USER_INTERNAL_KEY, ITEM_INTERNAL_KEY]).astype(str)
    assert test_interactions_df.equals(res_df)


def test_recommend_rated_item_with_sample_data():
    normal_params, port_params = default_params()
    port_params['training_interactions'] = None
    res_df = ScoreWideAndDeepRecommenderModule(**normal_params).run(**port_params)

    # verify column names
    assert all(res_df.columns == ['User', 'Item 1', 'Item 2', 'Item 3'])

    test_interactions_df = port_params['test_interactions'].preprocess().df
    test_interactions_df = test_interactions_df.iloc[:, [INTERACTIONS_USER_COL, INTERACTIONS_ITEM_COL]]
    test_interactions_df = test_interactions_df.rename(
        columns=dict(zip(test_interactions_df.columns, [USER_INTERNAL_KEY, ITEM_INTERNAL_KEY])))
    test_user_rated_items_group = test_interactions_df.groupby(by=USER_INTERNAL_KEY)
    test_user_rated_items = test_user_rated_items_group[ITEM_INTERNAL_KEY].apply(lambda x: list(x))[
        test_user_rated_items_group.size() >= normal_params['min_recommendation_pool_size']]
    test_user_rated_items = test_user_rated_items.sort_index()
    test_valid_users = test_user_rated_items.index.values
    test_user_rated_items = test_user_rated_items.values

    res_df = res_df.sort_values(USER_INTERNAL_KEY)
    scored_valid_users = res_df[USER_INTERNAL_KEY].values
    scored_recommend_items = res_df.drop(columns=USER_INTERNAL_KEY).values
    scored_recommend_items = [[item for item in items if not pd.isnull(item)] for items in scored_recommend_items]

    # verify valid users are as expected
    assert np.array_equal(test_valid_users, scored_valid_users)
    # verify recommend items are contained in rated items
    boolean_list = [set(scored_recommend_items[i]).issubset(set(test_user_rated_items[i])) for i in
                    range(len(scored_valid_users))]
    assert all(boolean_list)


def test_recommend_all_items_with_sample_data():
    normal_params, port_params = default_params()
    port_params['training_interactions'] = None
    normal_params['recommended_item_selection'] = RecommendedItemSelection.FromAllItems
    res_df = ScoreWideAndDeepRecommenderModule(**normal_params).run(**port_params)

    # verify column names
    assert all(res_df.columns == ['User', 'Item 1', 'Item 2', 'Item 3'])

    test_interactions_df = port_params['test_interactions'].preprocess().df
    test_users = np.sort(test_interactions_df.iloc[:, INTERACTIONS_USER_COL].unique())
    scored_users = np.sort(res_df.loc[:, USER_INTERNAL_KEY].values)

    # verify scored users are the same as test users
    assert np.array_equal(test_users, scored_users)
    # verify recommended item counts
    assert res_df.shape[1] == normal_params['max_recommended_item_count'] + 1


def test_recommend_unrated_items_with_sample_data():
    normal_params, port_params = default_params()
    normal_params['recommended_item_selection'] = RecommendedItemSelection.FromUnratedItems
    res_df = ScoreWideAndDeepRecommenderModule(**normal_params).run(**port_params)

    # verify column names
    assert all(res_df.columns == ['User', 'Item 1', 'Item 2', 'Item 3'])

    training_interactions_df = port_params['training_interactions'].preprocess().df
    training_interactions_df = training_interactions_df.rename(
        columns=dict(zip(training_interactions_df.columns, [USER_INTERNAL_KEY, ITEM_INTERNAL_KEY])))
    rated_items = training_interactions_df.groupby(USER_INTERNAL_KEY)[ITEM_INTERNAL_KEY].apply(list)

    test_interactions_df = port_params['test_interactions'].preprocess().df
    test_interactions_df = test_interactions_df.iloc[:, [INTERACTIONS_USER_COL]]
    test_interactions_df = test_interactions_df.rename(
        columns=dict(zip(test_interactions_df.columns, [USER_INTERNAL_KEY, ITEM_INTERNAL_KEY])))
    test_interactions_df = test_interactions_df.sort_values([USER_INTERNAL_KEY])
    test_users = test_interactions_df[USER_INTERNAL_KEY].unique()
    rated_items = rated_items[test_users].apply(lambda x: [] if type(x) != list and math.isnan(x) else x)

    res_df = res_df.sort_values(USER_INTERNAL_KEY)
    scored_users = res_df[USER_INTERNAL_KEY].values
    scored_recommended_items = res_df.drop(columns=USER_INTERNAL_KEY).values

    # verify users are as expected
    assert np.array_equal(test_users, scored_users)
    # verify recommended items are not in rated items
    boolean_list = [set(scored_recommended_items[i]).isdisjoint(set(rated_items[test_users[i]])) for i in
                    range(len(test_users))]
    assert all(boolean_list)
