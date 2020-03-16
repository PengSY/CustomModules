import pandas as pd
from azureml.studio.internal.error import ErrorMapping
from azureml.studio.core.logger import module_logger
from azureml.designer.modules.recommenders.dnn.wide_and_deep.common.wide_n_deep_model import WideNDeepModel
from azureml.designer.modules.recommenders.dnn.common.dataset import Dataset, TransactionDataset, FeatureDataset
from azureml.designer.modules.recommenders.dnn.wide_and_deep.score. \
    base_recommender_scorer import BaseRecommenderScorer
from azureml.designer.modules.recommenders.dnn.common.constants import USER_INTERNAL_KEY, TRANSACTIONS_RATING_COL


class RecommendRatedItemScorer(BaseRecommenderScorer):
    def _validate_parameters(self, learner: WideNDeepModel, test_data: Dataset, user_features: FeatureDataset = None,
                             item_features: FeatureDataset = None, **kwargs):
        super()._validate_parameters(learner, test_data, user_features=user_features, item_features=item_features)
        ErrorMapping.verify_number_of_columns_greater_than_or_equal_to(curr_column_count=test_data.column_size,
                                                                       required_column_count=2,
                                                                       arg_name=test_data.name)

    def score(self, learner: WideNDeepModel, test_transactions: TransactionDataset,
              user_features: FeatureDataset = None,
              item_features: FeatureDataset = None, **kwargs):
        module_logger.info("Recommendation task: Recommend items from rated item.")
        super().score(learner, test_transactions, user_features, item_features, **kwargs)
        max_recommended_item_count = kwargs["max_recommended_item_count"]
        min_recommendation_pool_size = kwargs["min_recommendation_pool_size"]
        return_ratings = kwargs["return_ratings"]
        transactions_df = test_transactions.df.iloc[:, :TRANSACTIONS_RATING_COL].copy()
        transactions_df = transactions_df.iloc[(~transactions_df.duplicated()).values, :]
        transactions_df = transactions_df.rename(columns=dict(zip(transactions_df.columns, [USER_INTERNAL_KEY])))
        user_group_size = transactions_df.groupby(USER_INTERNAL_KEY).size()
        valid_users_df = pd.DataFrame(
            {USER_INTERNAL_KEY: user_group_size.index[user_group_size >= min_recommendation_pool_size]})
        transactions_df = pd.merge(left=transactions_df, right=valid_users_df, how='inner')
        transactions = TransactionDataset(transactions_df)
        recommendations = self._recommend(learner, transactions=transactions, K=max_recommended_item_count,
                                          user_features=user_features, item_features=item_features)
        return self._format_recommendations(recommendations, return_ratings, K=max_recommended_item_count)
