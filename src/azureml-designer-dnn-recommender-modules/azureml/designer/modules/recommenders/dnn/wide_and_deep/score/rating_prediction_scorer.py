from azureml.studio.internal.error import ErrorMapping
from azureml.studio.core.logger import module_logger
from azureml.designer.modules.recommenders.dnn.common.dataset import Dataset, TransactionDataset, FeatureDataset
from azureml.designer.modules.recommenders.dnn.wide_and_deep.common.wide_n_deep_model import WideNDeepModel
from azureml.designer.modules.recommenders.dnn.wide_and_deep.score. \
    base_recommender_scorer import BaseRecommenderScorer
from azureml.designer.modules.recommenders.dnn.common.constants import TRANSACTIONS_RATING_COL


class RatingPredictionScorer(BaseRecommenderScorer):
    def _validate_parameters(self, learner: WideNDeepModel, test_data: Dataset, user_features: FeatureDataset = None,
                             item_features: FeatureDataset = None, **kwargs):
        super()._validate_parameters(learner, test_data, user_features=user_features, item_features=item_features)
        ErrorMapping.verify_number_of_columns_greater_than_or_equal_to(curr_column_count=test_data.column_size,
                                                                       required_column_count=2,
                                                                       arg_name=test_data.name)

    def score(self, learner: WideNDeepModel, test_transactions: TransactionDataset,
              user_features: FeatureDataset = None,
              item_features: FeatureDataset = None, **kwargs):
        module_logger.info("Recommendation task: Predict rating for user-item pairs.")
        super().score(learner, test_transactions, user_features, item_features, **kwargs)
        test_transactions_df = test_transactions.df.iloc[:, :TRANSACTIONS_RATING_COL].copy()
        test_transactions_df = test_transactions_df.iloc[(~test_transactions_df.duplicated()).values, :]
        test_transactions = TransactionDataset(test_transactions_df, name=test_transactions.name)
        return self._predict(learner, test_transactions, user_features=user_features, item_features=item_features)
