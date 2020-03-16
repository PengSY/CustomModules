from azureml.studio.internal.error import ErrorMapping, MoreThanOneRatingError, DuplicateFeatureDefinitionError, \
    InvalidDatasetError, InvalidColumnTypeError
from azureml.studio.core.data_frame_schema import ColumnTypeName
from azureml.designer.modules.recommenders.dnn.common.constants import TRANSACTIONS_RATING_COL, TRANSACTIONS_USER_COL, \
    TRANSACTIONS_ITEM_COL
from azureml.designer.modules.recommenders.dnn.common.entry_param import IntTuple, Boolean
from azureml.designer.modules.recommenders.dnn.common.dataset import TransactionDataset, FeatureDataset
from azureml.designer.modules.recommenders.dnn.common.entry_utils import params_loader
from azureml.designer.modules.recommenders.dnn.wide_and_deep.common.preprocess import preprocess_features, \
    preprocess_transactions
from azureml.designer.modules.recommenders.dnn.common.feature_builder import FeatureBuilder
from azureml.designer.modules.recommenders.dnn.wide_and_deep.common.wide_n_deep_model import WideNDeepModel, \
    WideNDeepModelHyperParams, OptimizerSelection, ActivationFnSelection
from azureml.designer.modules.recommenders.dnn.common.constants import USER_INTERNAL_KEY, ITEM_INTERNAL_KEY


class TrainWideAndDeepRecommenderModule:
    @staticmethod
    def _validate_features_column_type(dataset: FeatureDataset):
        for col in dataset.columns:
            if dataset.get_column_type(col) == ColumnTypeName.NAN:
                ErrorMapping.throw(InvalidColumnTypeError(col_type=dataset.get_column_type(col),
                                                          col_name=col,
                                                          arg_name=dataset.name))

    @staticmethod
    def _validate_feature_dataset(dataset: FeatureDataset):
        ErrorMapping.verify_number_of_columns_greater_than_or_equal_to(curr_column_count=dataset.column_size,
                                                                       required_column_count=2,
                                                                       arg_name=dataset.name)
        ErrorMapping.verify_number_of_rows_greater_than_or_equal_to(curr_row_count=dataset.row_size,
                                                                    required_row_count=1,
                                                                    arg_name=dataset.name)
        TrainWideAndDeepRecommenderModule._validate_features_column_type(dataset)

    @staticmethod
    def _validate_datasets(transactions: TransactionDataset, user_features: FeatureDataset = None,
                           item_features: FeatureDataset = None):
        ErrorMapping.verify_number_of_columns_equal_to(curr_column_count=transactions.column_size,
                                                       required_column_count=3,
                                                       arg_name=transactions.name)
        ErrorMapping.verify_number_of_rows_greater_than_or_equal_to(curr_row_count=transactions.row_size,
                                                                    required_row_count=1,
                                                                    arg_name=transactions.name)
        ErrorMapping.verify_element_type(type_=transactions.get_column_type(TRANSACTIONS_RATING_COL),
                                         expected_type=ColumnTypeName.NUMERIC,
                                         column_name=transactions.ratings.name,
                                         arg_name=transactions.name)
        if user_features is not None:
            TrainWideAndDeepRecommenderModule._validate_feature_dataset(user_features)
        if item_features is not None:
            TrainWideAndDeepRecommenderModule._validate_feature_dataset(item_features)

    @staticmethod
    def _preprocess(transactions: TransactionDataset, user_features: FeatureDataset, item_features: FeatureDataset):
        # preprocess transactions data
        transactions = preprocess_transactions(transactions)

        # preprocess user features
        user_features = preprocess_features(user_features) if user_features is not None else None
        item_features = preprocess_features(item_features) if item_features is not None else None
        TrainWideAndDeepRecommenderModule._validate_preprocessed_dataset(transactions, user_features=user_features,
                                                                         item_features=item_features)
        return transactions, user_features, item_features

    @staticmethod
    def _validate_preprocessed_dataset(transactions: TransactionDataset, user_features: FeatureDataset,
                                       item_features: FeatureDataset):
        if transactions.row_size <= 0:
            ErrorMapping.throw(
                InvalidDatasetError(dataset1=transactions.name, reason=f"dataset does not have any valid samples"))
        if transactions.df.duplicated(
                subset=transactions.columns[[TRANSACTIONS_USER_COL, TRANSACTIONS_ITEM_COL]]).any():
            ErrorMapping.throw(MoreThanOneRatingError())

        if user_features is not None and any(user_features.df.duplicated(subset=user_features.ids.name)):
            ErrorMapping.throw(DuplicateFeatureDefinitionError())
        if item_features is not None and any(item_features.df.duplicated(subset=item_features.ids.name)):
            ErrorMapping.throw(DuplicateFeatureDefinitionError())

    @staticmethod
    def set_inputs_name(transactions: TransactionDataset, user_features: FeatureDataset = None,
                        item_features: FeatureDataset = None):
        _TRANSACTIONS_NAME = "Training dataset of user-item-rating triples"
        _USER_FEATURES_NAME = "User features"
        _ITEM_FEATURES_NAME = "Item features"
        if transactions is not None:
            transactions.name = _TRANSACTIONS_NAME
        else:
            ErrorMapping.verify_not_null_or_empty(x=transactions, name=_TRANSACTIONS_NAME)
        if user_features is not None:
            user_features.name = _USER_FEATURES_NAME
        if item_features is not None:
            item_features.name = _ITEM_FEATURES_NAME

    @params_loader
    def run(self,
            transactions: TransactionDataset,
            user_features: FeatureDataset,
            item_features: FeatureDataset,
            epochs: int,
            batch_size: int,
            wide_optimizer: OptimizerSelection,
            wide_lr: float,
            crossed_dim: int,
            deep_optimizer: OptimizerSelection,
            deep_lr: float,
            user_dim: int,
            item_dim: int,
            embed_dim: int,
            hidden_units: IntTuple,
            activation_fn: ActivationFnSelection,
            dropout: float,
            batch_norm: Boolean,
            model_dir: str):
        self.set_inputs_name(transactions, user_features=user_features, item_features=item_features)
        self._validate_datasets(transactions, user_features=user_features, item_features=item_features)
        self._preprocess(transactions, user_features=user_features, item_features=item_features)

        hyper_params = WideNDeepModelHyperParams(epochs=epochs, batch_size=batch_size,
                                                 wide_optimizer=wide_optimizer, wide_lr=wide_lr,
                                                 deep_optimizer=deep_optimizer, deep_lr=deep_lr,
                                                 hidden_units=hidden_units, activation_fn=activation_fn,
                                                 dropout=dropout, batch_norm=batch_norm, crossed_dim=crossed_dim,
                                                 user_dim=user_dim, item_dim=item_dim,
                                                 embed_dim=embed_dim)
        user_feature_builder = FeatureBuilder(ids=transactions.users, id_key=USER_INTERNAL_KEY,
                                              features=user_features, feat_key_suffix='user_feature')
        item_feature_builder = FeatureBuilder(ids=transactions.items, id_key=ITEM_INTERNAL_KEY,
                                              features=item_features, feat_key_suffix='item_feature')
        model = WideNDeepModel(hyper_params=hyper_params, save_dir=model_dir, user_feature_builder=user_feature_builder,
                               item_feature_builder=item_feature_builder)
        model.train(transactions=transactions)
        model.save()
