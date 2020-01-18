from azureml.studio.internal.error import ErrorMapping
from azureml.studio.core.logger import module_logger
from azureml.designer.modules.recommenders.dnn.common.dataset import Dataset, InteractionDataset, FeatureDataset
from azureml.designer.modules.recommenders.dnn.common.wide_and_deep_model import WideAndDeepModel
from azureml.designer.modules.recommenders.dnn.wide_and_deep.score. \
    base_recommender_scorer import BaseRecommenderScorer
from azureml.designer.modules.recommenders.dnn.common.constants import INTERACTIONS_RATING_COL


class RatingPredictionScorer(BaseRecommenderScorer):
    def _validate_parameters(self, test_data: Dataset, user_features: FeatureDataset = None,
                             item_features: FeatureDataset = None, **kwargs):
        super()._validate_parameters(test_data, user_features=user_features, item_features=item_features)
        ErrorMapping.verify_number_of_columns_greater_than_or_equal_to(curr_column_count=test_data.column_size,
                                                                       required_column_count=2,
                                                                       arg_name=test_data.name)

    def score(self, learner: WideAndDeepModel, test_interactions: InteractionDataset,
              user_features: FeatureDataset = None,
              item_features: FeatureDataset = None, **kwargs):
        module_logger.info(f"Start to generate ratings.")
        super().score(learner, test_interactions, user_features, item_features, **kwargs)
        test_interactions_df = test_interactions.df.iloc[:, :INTERACTIONS_RATING_COL].copy()
        test_interactions_df = test_interactions_df.iloc[(~test_interactions_df.duplicated()).values, :]
        test_interactions = InteractionDataset(test_interactions_df, name=test_interactions.name)
        return self._predict(learner, test_interactions, user_features=user_features, item_features=item_features)
