import pandas as pd
from azureml.studio.internal.error import ErrorMapping
from azureml.studio.core.logger import module_logger
from azureml.designer.modules.recommenders.dnn.wide_and_deep.common.wide_and_deep_model import WideAndDeepModel
from azureml.designer.modules.recommenders.dnn.common.dataset import Dataset, InteractionDataset, FeatureDataset
from azureml.designer.modules.recommenders.dnn.wide_and_deep.score. \
    base_recommender_scorer import BaseRecommenderScorer
from azureml.designer.modules.recommenders.dnn.common.constants import USER_INTERNAL_KEY, INTERACTIONS_RATING_COL


class RecommendRatedItemScorer(BaseRecommenderScorer):
    def _validate_parameters(self, learner: WideAndDeepModel, test_data: Dataset, user_features: FeatureDataset = None,
                             item_features: FeatureDataset = None, **kwargs):
        super()._validate_parameters(learner, test_data, user_features=user_features, item_features=item_features)
        ErrorMapping.verify_number_of_columns_greater_than_or_equal_to(curr_column_count=test_data.column_size,
                                                                       required_column_count=2,
                                                                       arg_name=test_data.name)

    def score(self, learner: WideAndDeepModel, test_interactions: InteractionDataset,
              user_features: FeatureDataset = None,
              item_features: FeatureDataset = None, **kwargs):
        module_logger.info(f"Start to generate recommendations from rated items.")
        super().score(learner, test_interactions, user_features, item_features, **kwargs)
        max_recommended_item_count = kwargs["max_recommended_item_count"]
        min_recommendation_pool_size = kwargs["min_recommendation_pool_size"]
        return_ratings = kwargs["return_ratings"]
        interactions_df = test_interactions.df.iloc[:, :INTERACTIONS_RATING_COL].copy()
        interactions_df = interactions_df.iloc[(~interactions_df.duplicated()).values, :]
        interactions_df = interactions_df.rename(columns=dict(zip(interactions_df.columns, [USER_INTERNAL_KEY])))
        user_group_size = interactions_df.groupby(USER_INTERNAL_KEY).size()
        valid_users_df = pd.DataFrame(
            {USER_INTERNAL_KEY: user_group_size.index[user_group_size >= min_recommendation_pool_size]})
        interactions_df = pd.merge(left=interactions_df, right=valid_users_df, how='inner')
        interactions = InteractionDataset(interactions_df)
        recommendations = self._recommend(learner, interactions=interactions, K=max_recommended_item_count,
                                          user_features=user_features, item_features=item_features)
        return self._format_recommendations(recommendations, return_ratings, K=max_recommended_item_count)
