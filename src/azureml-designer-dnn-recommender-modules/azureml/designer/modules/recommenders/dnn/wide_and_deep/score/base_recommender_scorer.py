import numpy as np
import pandas as pd
from abc import abstractmethod
from azureml.studio.internal.error import ErrorMapping, DuplicateFeatureDefinitionError, InvalidColumnTypeError
from azureml.studio.core.data_frame_schema import ColumnTypeName
from azureml.designer.modules.recommenders.dnn.common.dataset import Dataset, TransactionDataset, FeatureDataset
from azureml.designer.modules.recommenders.dnn.wide_and_deep.common.wide_n_deep_model import WideNDeepModel
from azureml.designer.modules.recommenders.dnn.common.constants import USER_INTERNAL_KEY, ITEM_INTERNAL_KEY, \
    RATING_INTERNAL_KEY, TRANSACTIONS_RATING_COL
from azureml.designer.modules.recommenders.dnn.wide_and_deep.common.preprocess import preprocess_features, \
    preprocess_transactions


class BaseRecommenderScorer:
    def _validate_parameters(self, learner: WideNDeepModel, test_data: Dataset, user_features: FeatureDataset = None,
                             item_features: FeatureDataset = None, **kwargs):
        ErrorMapping.verify_not_null_or_empty(x=learner, name=WideNDeepModel.MODEL_NAME)
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

    @staticmethod
    def _validate_features_type(dataset: FeatureDataset):
        for col in dataset.columns:
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
    def _preprocess(transactions: TransactionDataset, user_features: FeatureDataset = None,
                    item_features: FeatureDataset = None, training_transactions: TransactionDataset = None):
        transactions = preprocess_transactions(transactions)
        user_features = preprocess_features(user_features) if user_features is not None else None
        item_features = preprocess_features(item_features) if item_features is not None else None
        training_transactions = (
            preprocess_transactions(training_transactions) if training_transactions is not None else None
        )

        BaseRecommenderScorer._validate_preprocessed_dataset(user_features=user_features, item_features=item_features)

        return transactions, user_features, item_features, training_transactions

    @staticmethod
    def _validate_preprocessed_dataset(user_features: FeatureDataset, item_features: FeatureDataset):
        if user_features is not None and any(user_features.df.duplicated(subset=user_features.ids.name)):
            ErrorMapping.throw(DuplicateFeatureDefinitionError())
        if item_features is not None and any(item_features.df.duplicated(subset=item_features.ids.name)):
            ErrorMapping.throw(DuplicateFeatureDefinitionError())

    @abstractmethod
    def score(self, learner: WideNDeepModel, test_transactions: TransactionDataset,
              user_features: FeatureDataset = None,
              item_features: FeatureDataset = None, **kwargs):
        self._validate_parameters(learner=learner, test_data=test_transactions, user_features=user_features,
                                  item_features=item_features, **kwargs)
        training_transactions = kwargs['training_transactions']
        self._preprocess(transactions=test_transactions, user_features=user_features, item_features=item_features,
                         training_transactions=training_transactions)

    def _predict(self, learner: WideNDeepModel, transactions: TransactionDataset,
                 user_features: FeatureDataset = None,
                 item_features: FeatureDataset = None):
        learner.update_feature_builders(user_features=user_features, item_features=item_features)
        transactions = TransactionDataset(df=transactions.df.iloc[:, :TRANSACTIONS_RATING_COL])
        predictions = learner.predict(transactions)
        result_df = transactions.df.copy()
        result_df = result_df.rename(columns=dict(zip(result_df.columns, [USER_INTERNAL_KEY, ITEM_INTERNAL_KEY])))
        result_df[RATING_INTERNAL_KEY] = predictions
        return result_df

    def _recommend(self, learner: WideNDeepModel, transactions: TransactionDataset, K: int,
                   user_features: FeatureDataset = None, item_features: FeatureDataset = None):
        predict_df = self._predict(learner, transactions, user_features=user_features, item_features=item_features)
        predict_df = predict_df.sort_values(by=[USER_INTERNAL_KEY, RATING_INTERNAL_KEY], ascending=False)
        topK_items = predict_df.groupby(USER_INTERNAL_KEY)[ITEM_INTERNAL_KEY].apply(
            lambda x: (list(x) + [None] * K)[:K])
        topK_ratings = predict_df.groupby(USER_INTERNAL_KEY)[RATING_INTERNAL_KEY].apply(
            lambda x: (list(x) + [0] * K)[:K])
        return pd.DataFrame(
            {USER_INTERNAL_KEY: topK_items.index, ITEM_INTERNAL_KEY: topK_items.values,
             RATING_INTERNAL_KEY: topK_ratings.values})

    def _format_recommendations(self, recommendations: pd.DataFrame, return_ratings: bool, K: int):
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
