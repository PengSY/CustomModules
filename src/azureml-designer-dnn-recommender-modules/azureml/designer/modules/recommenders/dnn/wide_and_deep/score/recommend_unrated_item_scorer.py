import pandas as pd
from azureml.studio.core.logger import module_logger
from azureml.studio.internal.error import ErrorMapping
from azureml.designer.modules.recommenders.dnn.common.constants import INTERACTIONS_USER_COL, USER_INTERNAL_KEY, \
    INTERACTIONS_ITEM_COL, ITEM_INTERNAL_KEY
from azureml.designer.modules.recommenders.dnn.common.dataset import Dataset, InteractionDataset, FeatureDataset
from azureml.designer.modules.recommenders.dnn.common.wide_and_deep_model import WideAndDeepModel
from azureml.designer.modules.recommenders.dnn.wide_and_deep.score. \
    base_recommender_scorer import BaseRecommenderScorer


class RecommendUnratedItemScorer(BaseRecommenderScorer):
    def _validate_parameters(self, test_data: Dataset, user_features: FeatureDataset = None,
                             item_features: FeatureDataset = None, **kwargs):
        training_interactions = kwargs["training_interactions"]
        super()._validate_parameters(test_data, user_features=user_features, item_features=item_features)
        ErrorMapping.verify_not_null_or_empty(training_interactions, name="Training data")
        ErrorMapping.verify_number_of_rows_greater_than_or_equal_to(curr_row_count=training_interactions.row_size,
                                                                    required_row_count=1,
                                                                    arg_name=training_interactions.name)
        ErrorMapping.verify_number_of_columns_less_than_or_equal_to(curr_column_count=training_interactions.column_size,
                                                                    required_column_count=3,
                                                                    arg_name=training_interactions.name)
        ErrorMapping.verify_number_of_columns_greater_than_or_equal_to(
            curr_column_count=training_interactions.column_size,
            required_column_count=2,
            arg_name=training_interactions.name)
        ErrorMapping.verify_number_of_columns_greater_than_or_equal_to(curr_column_count=test_data.column_size,
                                                                       required_column_count=2,
                                                                       arg_name=test_data.name)

    def score(self, learner: WideAndDeepModel, test_interactions: InteractionDataset,
              user_features: FeatureDataset = None,
              item_features: FeatureDataset = None, **kwargs):
        module_logger.info(f"Start to generate recommendations from unrated items.")
        super().score(learner, test_interactions, user_features, item_features, **kwargs)
        DUMMY_KEY = 'key'
        DUMMY_VALUE = 0
        INDEX_COL = 'index'
        max_recommended_item_count = kwargs["max_recommended_item_count"]
        return_ratings = kwargs["return_ratings"]
        training_interactions = kwargs["training_interactions"]
        all_items = learner.feature_builder.item_vocab
        all_items = pd.DataFrame({DUMMY_KEY: DUMMY_VALUE, ITEM_INTERNAL_KEY: all_items})
        test_interactions_df = test_interactions.df.copy()
        training_interactions_df = training_interactions.df.copy()
        test_interactions_df = test_interactions_df.rename(
            columns={test_interactions.columns[INTERACTIONS_USER_COL]: USER_INTERNAL_KEY})
        training_interactions_df = training_interactions_df.rename(
            columns={training_interactions_df.columns[INTERACTIONS_USER_COL]: USER_INTERNAL_KEY,
                     training_interactions_df.columns[INTERACTIONS_ITEM_COL]: ITEM_INTERNAL_KEY})
        users = test_interactions_df[USER_INTERNAL_KEY].to_frame()
        users[DUMMY_KEY] = DUMMY_VALUE
        user_items = pd.merge(users, all_items, on=DUMMY_KEY)[[USER_INTERNAL_KEY, ITEM_INTERNAL_KEY]]
        common_pairs = pd.merge(user_items.reset_index(), training_interactions_df,
                                on=[USER_INTERNAL_KEY, ITEM_INTERNAL_KEY])
        index = set(user_items.index) - set(common_pairs.pop(INDEX_COL))
        test_interactions = InteractionDataset(user_items.loc[index])
        recommendations = self._recommend(learner, interactions=test_interactions, K=max_recommended_item_count,
                                          user_features=user_features, item_features=item_features)
        return self._format_recommendations(recommendations, return_ratings, K=max_recommended_item_count)
