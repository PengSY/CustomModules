import pandas as pd
from azureml.studio.core.logger import module_logger
from azureml.studio.internal.error import ErrorMapping
from azureml.designer.modules.recommenders.dnn.common.constants import TRANSACTIONS_USER_COL, USER_INTERNAL_KEY, \
    TRANSACTIONS_ITEM_COL, ITEM_INTERNAL_KEY
from azureml.designer.modules.recommenders.dnn.common.dataset import Dataset, TransactionDataset, FeatureDataset
from azureml.designer.modules.recommenders.dnn.wide_and_deep.common.wide_n_deep_model import WideNDeepModel
from azureml.designer.modules.recommenders.dnn.wide_and_deep.score. \
    base_recommender_scorer import BaseRecommenderScorer


class RecommendUnratedItemScorer(BaseRecommenderScorer):
    def _validate_parameters(self, learner: WideNDeepModel, test_data: Dataset, user_features: FeatureDataset = None,
                             item_features: FeatureDataset = None, **kwargs):
        training_transactions = kwargs["training_transactions"]
        super()._validate_parameters(learner, test_data, user_features=user_features, item_features=item_features)
        ErrorMapping.verify_not_null_or_empty(training_transactions, name="Training data")
        ErrorMapping.verify_number_of_rows_greater_than_or_equal_to(curr_row_count=training_transactions.row_size,
                                                                    required_row_count=1,
                                                                    arg_name=training_transactions.name)
        ErrorMapping.verify_number_of_columns_less_than_or_equal_to(curr_column_count=training_transactions.column_size,
                                                                    required_column_count=3,
                                                                    arg_name=training_transactions.name)
        ErrorMapping.verify_number_of_columns_greater_than_or_equal_to(
            curr_column_count=training_transactions.column_size,
            required_column_count=2,
            arg_name=training_transactions.name)
        ErrorMapping.verify_number_of_columns_greater_than_or_equal_to(curr_column_count=test_data.column_size,
                                                                       required_column_count=2,
                                                                       arg_name=test_data.name)

    def score(self, learner: WideNDeepModel, test_transactions: TransactionDataset,
              user_features: FeatureDataset = None,
              item_features: FeatureDataset = None, **kwargs):
        module_logger.info("Recommendation task: Recommend items from unrated item.")
        super().score(learner, test_transactions, user_features, item_features, **kwargs)
        DUMMY_KEY = 'key'
        DUMMY_VALUE = 0
        INDEX_COL = 'index'
        max_recommended_item_count = kwargs["max_recommended_item_count"]
        return_ratings = kwargs["return_ratings"]
        training_transactions = kwargs["training_transactions"]
        all_items = learner.item_feature_builder.id_vocab
        all_items = pd.DataFrame({DUMMY_KEY: DUMMY_VALUE, ITEM_INTERNAL_KEY: all_items})
        test_transactions_df = test_transactions.df.copy()
        training_transactions_df = training_transactions.df.copy()
        test_transactions_df = test_transactions_df.rename(
            columns={test_transactions.columns[TRANSACTIONS_USER_COL]: USER_INTERNAL_KEY})
        training_transactions_df = training_transactions_df.rename(
            columns={training_transactions_df.columns[TRANSACTIONS_USER_COL]: USER_INTERNAL_KEY,
                     training_transactions_df.columns[TRANSACTIONS_ITEM_COL]: ITEM_INTERNAL_KEY})
        users = test_transactions_df[USER_INTERNAL_KEY].unique()
        module_logger.info(f"Get {len(users)} unique users.")
        users = pd.DataFrame({USER_INTERNAL_KEY: users, DUMMY_KEY: DUMMY_VALUE})
        user_items = pd.merge(users, all_items, on=DUMMY_KEY)[[USER_INTERNAL_KEY, ITEM_INTERNAL_KEY]]
        common_pairs = pd.merge(user_items.reset_index(), training_transactions_df,
                                on=[USER_INTERNAL_KEY, ITEM_INTERNAL_KEY])
        index = set(user_items.index) - set(common_pairs.pop(INDEX_COL))
        test_transactions = TransactionDataset(user_items.loc[index])
        recommendations = self._recommend(learner, transactions=test_transactions, K=max_recommended_item_count,
                                          user_features=user_features, item_features=item_features)
        return self._format_recommendations(recommendations, return_ratings, K=max_recommended_item_count)
