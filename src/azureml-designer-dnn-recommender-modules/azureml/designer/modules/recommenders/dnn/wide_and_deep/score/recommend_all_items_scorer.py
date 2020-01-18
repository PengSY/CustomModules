import pandas as pd
from azureml.studio.core.logger import module_logger
from azureml.studio.internal.error import ErrorMapping
from azureml.designer.modules.recommenders.dnn.common.dataset import Dataset, InteractionDataset, FeatureDataset
from azureml.designer.modules.recommenders.dnn.common.wide_and_deep_model import WideAndDeepModel
from azureml.designer.modules.recommenders.dnn.common.constants import ITEM_INTERNAL_KEY, USER_INTERNAL_KEY, \
    INTERACTIONS_USER_COL
from azureml.designer.modules.recommenders.dnn.wide_and_deep.score. \
    base_recommender_scorer import BaseRecommenderScorer


class RecommendAllItemScorer(BaseRecommenderScorer):
    def _validate_parameters(self, test_data: Dataset, user_features: FeatureDataset = None,
                             item_features: FeatureDataset = None, **kwargs):
        super()._validate_parameters(test_data, user_features=user_features, item_features=item_features)
        ErrorMapping.verify_number_of_columns_greater_than_or_equal_to(curr_column_count=test_data.column_size,
                                                                       required_column_count=1,
                                                                       arg_name=test_data.name)

    def score(self, learner: WideAndDeepModel, test_interactions: InteractionDataset,
              user_features: FeatureDataset = None,
              item_features: FeatureDataset = None, **kwargs):
        module_logger.info(f"Start to generate recommendations from all items.")
        super().score(learner, test_interactions, user_features, item_features, **kwargs)
        max_recommended_item_count = kwargs["max_recommended_item_count"]
        return_ratings = kwargs["return_ratings"]
        all_items = learner.feature_builder.item_vocab
        test_interactions_df = test_interactions.df
        users = test_interactions_df.iloc[:, INTERACTIONS_USER_COL].unique()
        recommendations = []
        for user in users:
            u_interactions_df = pd.DataFrame({USER_INTERNAL_KEY: [user] * len(all_items)})
            u_interactions_df[ITEM_INTERNAL_KEY] = all_items
            u_interactions = InteractionDataset(u_interactions_df)
            u_recommendations = self._recommend(learner, interactions=u_interactions, K=max_recommended_item_count,
                                                user_features=user_features, item_features=item_features)
            recommendations.append(u_recommendations)
        recommendations = pd.concat(recommendations, axis=0)

        return self._format_recommendations(recommendations, return_ratings, K=max_recommended_item_count)
