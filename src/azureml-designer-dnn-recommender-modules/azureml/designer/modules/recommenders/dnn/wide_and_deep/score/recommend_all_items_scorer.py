import pandas as pd
from azureml.studio.core.logger import module_logger
from azureml.studio.internal.error import ErrorMapping
from azureml.designer.modules.recommenders.dnn.common.dataset import Dataset, TransactionDataset, FeatureDataset
from azureml.designer.modules.recommenders.dnn.wide_and_deep.common.wide_n_deep_model import WideNDeepModel
from azureml.designer.modules.recommenders.dnn.common.constants import ITEM_INTERNAL_KEY, USER_INTERNAL_KEY, \
    TRANSACTIONS_USER_COL
from azureml.designer.modules.recommenders.dnn.wide_and_deep.score. \
    base_recommender_scorer import BaseRecommenderScorer


class RecommendAllItemScorer(BaseRecommenderScorer):
    def _validate_parameters(self, learner: WideNDeepModel, test_data: Dataset, user_features: FeatureDataset = None,
                             item_features: FeatureDataset = None, **kwargs):
        super()._validate_parameters(learner, test_data, user_features=user_features, item_features=item_features)
        ErrorMapping.verify_number_of_columns_greater_than_or_equal_to(curr_column_count=test_data.column_size,
                                                                       required_column_count=1,
                                                                       arg_name=test_data.name)

    def score(self, learner: WideNDeepModel, test_transactions: TransactionDataset,
              user_features: FeatureDataset = None,
              item_features: FeatureDataset = None, **kwargs):
        module_logger.info("Recommendation task: Recommend items from all item.")
        super().score(learner, test_transactions, user_features, item_features, **kwargs)
        DUMMY_KEY = 'key'
        DUMMY_VALUE = 0
        max_recommended_item_count = kwargs["max_recommended_item_count"]
        return_ratings = kwargs["return_ratings"]
        all_items = learner.item_feature_builder.id_vocab
        all_items = pd.DataFrame({DUMMY_KEY: DUMMY_VALUE, ITEM_INTERNAL_KEY: all_items})
        test_transactions_df = test_transactions.df
        users = test_transactions_df.iloc[:, TRANSACTIONS_USER_COL].unique()
        module_logger.info(f"Get {len(users)} unique users.")
        users = pd.DataFrame({USER_INTERNAL_KEY: users, DUMMY_KEY: DUMMY_VALUE})
        test_transactions = TransactionDataset(
            pd.merge(users, all_items, on=DUMMY_KEY)[[USER_INTERNAL_KEY, ITEM_INTERNAL_KEY]])
        recommendations = self._recommend(learner, transactions=test_transactions, K=max_recommended_item_count,
                                          user_features=user_features, item_features=item_features)
        return self._format_recommendations(recommendations, return_ratings, K=max_recommended_item_count)
