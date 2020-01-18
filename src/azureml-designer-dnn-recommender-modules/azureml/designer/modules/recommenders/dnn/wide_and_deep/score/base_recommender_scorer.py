import numpy as np
import pandas as pd
from abc import abstractmethod
from azureml.studio.core.logger import module_logger
from azureml.studio.internal.error import ErrorMapping, DuplicateFeatureDefinitionError, InvalidColumnTypeError
from azureml.studio.core.data_frame_schema import ColumnTypeName
from azureml.designer.modules.recommenders.dnn.common.dataset import Dataset, InteractionDataset, FeatureDataset
from azureml.designer.modules.recommenders.dnn.common.wide_and_deep_model import WideAndDeepModel
from azureml.designer.modules.recommenders.dnn.common.constants import USER_INTERNAL_KEY, ITEM_INTERNAL_KEY, \
    RATING_INTERNAL_KEY, INTERACTIONS_RATING_COL, VALID_FEATURE_TYPE


class BaseRecommenderScorer:
    def _validate_parameters(self, test_data: Dataset, user_features: FeatureDataset = None,
                             item_features: FeatureDataset = None, **kwargs):
        ErrorMapping.verify_number_of_rows_greater_than_or_equal_to(curr_row_count=test_data.row_size,
                                                                    required_row_count=1,
                                                                    arg_name=test_data.name)
        ErrorMapping.verify_number_of_columns_less_than_or_equal_to(curr_column_count=test_data.column_size,
                                                                    required_column_count=3,
                                                                    arg_name=test_data.name)
        if user_features is not None:
            self._validate_feature_dataset(user_features)
        if item_features is not None:
            self._validate_feature_dataset(item_features)

    def _validate_features_type(self, dataset: FeatureDataset):
        id_col = dataset.ids.name
        for col in dataset.columns:
            if col != id_col and dataset.get_column_type(col) not in VALID_FEATURE_TYPE:
                ErrorMapping.verify_element_type(type_=dataset.get_column_type(col),
                                                 expected_type=' or '.join(VALID_FEATURE_TYPE),
                                                 column_name=col,
                                                 arg_name=dataset.name)
            else:
                if dataset.get_column_type(col) == ColumnTypeName.NAN:
                    ErrorMapping.throw(InvalidColumnTypeError(col_type=dataset.get_column_type(col),
                                                              col_name=col,
                                                              arg_name=dataset.name))

    def _validate_feature_dataset(self, dataset: FeatureDataset):
        ErrorMapping.verify_number_of_columns_greater_than_or_equal_to(curr_column_count=dataset.column_size,
                                                                       required_column_count=2,
                                                                       arg_name=dataset.name)
        ErrorMapping.verify_number_of_rows_greater_than_or_equal_to(curr_row_count=dataset.row_size,
                                                                    required_row_count=1,
                                                                    arg_name=dataset.name)
        self._validate_features_type(dataset)

    @staticmethod
    def _preprocess(interactions: InteractionDataset, user_features: FeatureDataset = None,
                    item_features: FeatureDataset = None, training_interactions: InteractionDataset = None):
        interactions = interactions.preprocess()
        if user_features is not None:
            user_features = user_features.preprocess_ids()
        if item_features is not None:
            item_features = item_features.preprocess_ids()
        if training_interactions is not None:
            training_interactions = training_interactions.preprocess()
        BaseRecommenderScorer._validate_preprocessed_dataset(user_features=user_features, item_features=item_features)

        return interactions, user_features, item_features, training_interactions

    @staticmethod
    def _validate_preprocessed_dataset(user_features: FeatureDataset, item_features: FeatureDataset):
        if user_features is not None and any(user_features.df.duplicated(subset=user_features.ids.name)):
            ErrorMapping.throw(DuplicateFeatureDefinitionError())
        if item_features is not None and any(item_features.df.duplicated(subset=item_features.ids.name)):
            ErrorMapping.throw(DuplicateFeatureDefinitionError())

    @abstractmethod
    def score(self, learner: WideAndDeepModel, test_interactions: InteractionDataset,
              user_features: FeatureDataset = None,
              item_features: FeatureDataset = None, **kwargs):
        self._validate_parameters(test_data=test_interactions, user_features=user_features, item_features=item_features,
                                  **kwargs)
        training_interactions = kwargs['training_interactions']
        self._preprocess(interactions=test_interactions, user_features=user_features, item_features=item_features,
                         training_interactions=training_interactions)
        pass

    def _predict(self, learner: WideAndDeepModel, interactions: InteractionDataset,
                 user_features: FeatureDataset = None,
                 item_features: FeatureDataset = None):
        module_logger.info(f"Base scorer _predict function.")
        interactions = InteractionDataset(df=interactions.df.iloc[:, :INTERACTIONS_RATING_COL])
        predictions = learner.predict(interactions, user_features=user_features, item_features=item_features)
        result_df = interactions.df.copy()
        result_df = result_df.rename(columns=dict(zip(result_df.columns, [USER_INTERNAL_KEY, ITEM_INTERNAL_KEY])))
        result_df[RATING_INTERNAL_KEY] = predictions
        return result_df

    def _recommend(self, learner: WideAndDeepModel, interactions: InteractionDataset, K: int,
                   user_features: FeatureDataset = None, item_features: FeatureDataset = None):
        module_logger.info(f"Base scorer _recommend function.")
        predict_df = self._predict(learner, interactions, user_features=user_features, item_features=item_features)
        predict_df = predict_df.sort_values(by=[USER_INTERNAL_KEY, RATING_INTERNAL_KEY])
        topK_items = predict_df.groupby(USER_INTERNAL_KEY)[ITEM_INTERNAL_KEY].apply(
            lambda x: (list(x) + [None] * K)[:K])
        topK_ratings = predict_df.groupby(USER_INTERNAL_KEY)[RATING_INTERNAL_KEY].apply(
            lambda x: (list(x) + [0] * K)[:K])
        return pd.DataFrame(
            {USER_INTERNAL_KEY: topK_items.index, ITEM_INTERNAL_KEY: topK_items.values,
             RATING_INTERNAL_KEY: topK_ratings.values})

    def _format_recommendations(self, recommendations: pd.DataFrame, return_ratings: bool, K: int):
        module_logger.info(f"Generate formatted recommendation result.")
        users = recommendations[USER_INTERNAL_KEY].values[:, np.newaxis]
        items = np.array(list(recommendations[ITEM_INTERNAL_KEY])).reshape([-1, K])
        ratings = np.array(list(recommendations[RATING_INTERNAL_KEY])).reshape([-1, K])
        recommendations = np.concatenate((users, items, ratings), axis=1)
        item_cols = [f"{ITEM_INTERNAL_KEY} {idx}" for idx in range(1, K + 1)]
        rating_cols = [f"{RATING_INTERNAL_KEY} {idx}" for idx in range(1, K + 1)]
        columns = [USER_INTERNAL_KEY] + item_cols + rating_cols
        recommendations = pd.DataFrame(recommendations, columns=columns)
        if return_ratings:
            columns[1::2] = item_cols
            columns[2::2] = rating_cols
        else:
            columns = columns[:K + 1]
        recommendations = recommendations[columns]
        return recommendations
