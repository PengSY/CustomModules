import os
import pickle
from azureml.studio.internal.error import ErrorMapping, MoreThanOneRatingError, DuplicateFeatureDefinitionError, \
    InvalidDatasetError, InvalidColumnTypeError
from azureml.studio.core.data_frame_schema import ColumnTypeName
from azureml.designer.modules.recommenders.dnn.common.utils import before_run
from azureml.designer.modules.recommenders.dnn.common.constants import INTERACTIONS_RATING_COL, INTERACTIONS_USER_COL, \
    INTERACTIONS_ITEM_COL, VALID_FEATURE_TYPE, MODEL_NAME
from azureml.designer.modules.recommenders.dnn.common.extend_types import IntTuple, DeepActivationSelection, \
    OptimizerSelection, Boolean
from azureml.designer.modules.recommenders.dnn.common.dataset import InteractionDataset, FeatureDataset, Dataset
from azureml.designer.modules.recommenders.dnn.common.wide_and_deep_model import WideAndDeepModel


class TrainWideAndDeepRecommenderModule:
    @staticmethod
    def _validate_features_column_type(dataset: FeatureDataset):
        id_col = dataset.ids.name
        for col in dataset.columns:
            if col != id_col:
                TrainWideAndDeepRecommenderModule._validate_column_type_in(dataset, VALID_FEATURE_TYPE, col)
            else:
                if dataset.get_column_type(col) == ColumnTypeName.NAN:
                    ErrorMapping.throw(InvalidColumnTypeError(col_type=dataset.get_column_type(col),
                                                              col_name=col,
                                                              arg_name=dataset.name))

    @staticmethod
    def _validate_column_type_in(dataset: Dataset, valid_types, column):
        if dataset.get_column_type(column) not in valid_types:
            ErrorMapping.verify_element_type(type_=dataset.get_column_type(column),
                                             expected_type=' or '.join(valid_types),
                                             column_name=column,
                                             arg_name=dataset.name)

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
    def _validate_datasets(interactions: InteractionDataset, user_features: FeatureDataset = None,
                           item_features: FeatureDataset = None):
        ErrorMapping.verify_number_of_columns_equal_to(curr_column_count=interactions.column_size,
                                                       required_column_count=3,
                                                       arg_name=interactions.name)
        ErrorMapping.verify_number_of_rows_greater_than_or_equal_to(curr_row_count=interactions.row_size,
                                                                    required_row_count=1,
                                                                    arg_name=interactions.name)
        ErrorMapping.verify_element_type(type_=interactions.get_column_type(INTERACTIONS_RATING_COL),
                                         expected_type=ColumnTypeName.NUMERIC,
                                         column_name=interactions.ratings.name,
                                         arg_name=interactions.name)
        if user_features is not None:
            TrainWideAndDeepRecommenderModule._validate_feature_dataset(user_features)
        if item_features is not None:
            TrainWideAndDeepRecommenderModule._validate_feature_dataset(item_features)

    @staticmethod
    def _preprocess(interactions: InteractionDataset, user_features: FeatureDataset, item_features: FeatureDataset):
        interactions = interactions.preprocess()
        user_features = user_features.preprocess_ids() if user_features is not None else None
        item_features = item_features.preprocess_ids() if item_features is not None else None
        TrainWideAndDeepRecommenderModule._validate_preprocessed_dataset(interactions, user_features=user_features,
                                                                         item_features=item_features)
        return interactions, user_features, item_features

    @staticmethod
    def _validate_preprocessed_dataset(interactions: InteractionDataset, user_features: FeatureDataset,
                                       item_features: FeatureDataset):
        if interactions.row_size <= 0:
            ErrorMapping.throw(
                InvalidDatasetError(dataset1=interactions.name, reason=f"dataset does not have any valid samples"))
        if interactions.df.duplicated(
                subset=interactions.columns[[INTERACTIONS_USER_COL, INTERACTIONS_ITEM_COL]]).any():
            ErrorMapping.throw(MoreThanOneRatingError())

        if user_features is not None and any(user_features.df.duplicated(subset=user_features.ids.name)):
            ErrorMapping.throw(DuplicateFeatureDefinitionError())
        if item_features is not None and any(item_features.df.duplicated(subset=item_features.ids.name)):
            ErrorMapping.throw(DuplicateFeatureDefinitionError())

    @staticmethod
    def dumper(model: WideAndDeepModel, model_dir):
        save_path = os.path.join(model_dir, MODEL_NAME + '.pkl')
        with open(save_path, "wb") as f:
            pickle.dump(model, f)

    @before_run
    def run(self,
            interactions: InteractionDataset,
            user_features: FeatureDataset,
            item_features: FeatureDataset,
            epochs: int,
            batch_size: int,
            wide_part_optimizer: OptimizerSelection,
            wide_learning_rate: float,
            crossed_dim: int,
            deep_part_optimizer: OptimizerSelection,
            deep_learning_rate: float,
            user_dim: int,
            item_dim: int,
            categorical_feature_dim: int,
            deep_hidden_units: IntTuple,
            deep_activation_fn: DeepActivationSelection,
            deep_dropout: float,
            batch_norm: Boolean,
            model_dir: str):
        self._validate_datasets(interactions, user_features=user_features,
                                item_features=item_features)
        self._preprocess(interactions, user_features=user_features,
                         item_features=item_features)
        model = WideAndDeepModel(epochs=epochs,
                                 batch_size=batch_size,
                                 wide_part_optimizer=wide_part_optimizer,
                                 wide_learning_rate=wide_learning_rate,
                                 deep_part_optimizer=deep_part_optimizer,
                                 deep_learning_rate=deep_learning_rate,
                                 deep_hidden_units=deep_hidden_units,
                                 deep_activation_fn=deep_activation_fn,
                                 deep_dropout=deep_dropout,
                                 batch_norm=batch_norm,
                                 crossed_dim=crossed_dim,
                                 user_dim=user_dim,
                                 item_dim=item_dim,
                                 categorical_feature_dim=categorical_feature_dim,
                                 model_dir=model_dir)
        model.train(interactions=interactions, user_features=user_features, item_features=item_features)
        self.dumper(model, model_dir)
