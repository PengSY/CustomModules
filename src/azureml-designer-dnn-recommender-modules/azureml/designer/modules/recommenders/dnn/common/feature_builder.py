from enum import Enum
import numpy as np
import pandas as pd
from azureml.studio.core.data_frame_schema import ColumnTypeName
from azureml.studio.internal.error import ColumnNotFoundError, ErrorMapping, InvalidDatasetError
from azureml.designer.modules.recommenders.dnn.common.dataset import WideDeepDataset, FeatureDataset


class FeatureGroup(Enum):
    UserFeature = "UserFeature"
    ItemFeature = "ItemFeature"


class FeatureMeta:
    def __init__(self, external_name, internal_name, feature_type, fillna_value=None, vocab=None, shape=(1,)):
        self.external_name = external_name
        self.internal_name = internal_name
        self.feature_type = feature_type
        self.fillna_value = fillna_value
        self.vocab = vocab
        self.shape = shape

    def is_categorical(self):
        return self.vocab is not None


class FeatureBuilder:
    def __init__(self):
        self.user_features = None
        self.item_features = None

        self.user_vocab = None
        self.item_vocab = None

        self.user_feature_metas = []
        self.item_feature_metas = []

    @staticmethod
    def _init_feature_metas(dataset: WideDeepDataset, features: FeatureDataset, feature_group: FeatureGroup):
        feature_metas = []
        if feature_group == FeatureGroup.UserFeature:
            internal_features_keys = dataset.features.columns[:features.column_size - 1]
        else:
            internal_features_keys = dataset.features.columns[-(features.column_size - 1):]
        external_features_keys = features.features.columns
        for external_key, internal_key in zip(external_features_keys, internal_features_keys):
            feature_type = features.get_column_type(external_key)
            feature_meta = FeatureMeta(external_name=external_key, internal_name=internal_key,
                                       feature_type=feature_type)
            if feature_meta.feature_type == ColumnTypeName.NUMERIC:
                feature_meta.fillna_value = dataset.features[feature_meta.internal_name].replace(
                    to_replace=[np.inf, -np.inf], value=np.nan).mean()
                if pd.isna(feature_meta.fillna_value):
                    FeatureBuilder._throw_invalid_features_error(dataset_name=features.name,
                                                                 feature_name=feature_meta.external_name)
            else:
                feature_meta.vocab = dataset.features[feature_meta.internal_name].dropna().astype(str).unique()
                if len(feature_meta.vocab) == 0:
                    FeatureBuilder._throw_invalid_features_error(dataset_name=features.name,
                                                                 feature_name=feature_meta.external_name)
                feature_meta.fillna_value = ''
            feature_metas.append(feature_meta)
        return feature_metas

    @staticmethod
    def _throw_invalid_features_error(dataset_name, feature_name):
        ErrorMapping.throw(InvalidDatasetError(dataset1=dataset_name,
                                               reason=f"dataset contains none valid {feature_name} feature"))

    def fit(self, dataset: WideDeepDataset, user_features: FeatureDataset = None, item_features: FeatureDataset = None):
        self.user_vocab = dataset.users.unique()
        self.item_vocab = dataset.items.unique()
        self.user_features = user_features
        self.item_features = item_features

        if user_features is not None:
            self.user_feature_metas = self._init_feature_metas(dataset=dataset, features=user_features,
                                                               feature_group=FeatureGroup.UserFeature)
        if item_features is not None:
            self.item_feature_metas = self._init_feature_metas(dataset=dataset, features=item_features,
                                                               feature_group=FeatureGroup.ItemFeature)
        return self

    def build(self, dataset: WideDeepDataset):
        dataset_df = dataset.df
        for feature_meta in self.user_feature_metas + self.item_feature_metas:
            if feature_meta.feature_type == ColumnTypeName.NUMERIC:
                dataset_df[feature_meta.internal_name] = dataset_df[feature_meta.internal_name].replace(
                    to_replace=[np.inf, -np.inf], value=np.nan).fillna(feature_meta.fillna_value)
            elif feature_meta.feature_type in [ColumnTypeName.BINARY, ColumnTypeName.CATEGORICAL]:
                dataset_df[feature_meta.internal_name] = dataset_df[feature_meta.internal_name].fillna(
                    feature_meta.fillna_value).astype(str)
            else:
                dataset_df[feature_meta.internal_name] = dataset_df[feature_meta.internal_name].fillna(
                    feature_meta.fillna_value)
        dataset.build_column_attributes()
        return dataset

    def update_features(self, user_features: FeatureDataset = None, item_features: FeatureDataset = None):
        if user_features is not None and self.user_features is not None:
            self._check_features_compatibility(features=user_features, feature_metas=self.user_feature_metas)
            self.user_features = self._get_updated_features(old_features=self.user_features,
                                                            curr_features=user_features)
        if item_features is not None and self.item_features is not None:
            self._check_features_compatibility(features=item_features, feature_metas=self.item_feature_metas)
            self.item_features = self._get_updated_features(old_features=self.item_features,
                                                            curr_features=item_features)

        return self

    @staticmethod
    def _check_features_compatibility(features: FeatureDataset, feature_metas):
        for feature_meta in feature_metas:
            feature_name = feature_meta.external_name
            if feature_name not in features.df:
                ErrorMapping.throw(
                    ColumnNotFoundError(column_id=feature_name, arg_name_missing_column=features.name))
            curr_type = features.get_column_type(feature_name)
            if curr_type != feature_meta.feature_type:
                ErrorMapping.verify_element_type(type_=curr_type, expected_type=feature_meta.feature_type,
                                                 column_name=feature_name,
                                                 arg_name=features.name)

    @staticmethod
    def _get_updated_features(old_features: FeatureDataset, curr_features: FeatureDataset):
        curr_features_df = curr_features.df
        curr_features_df = curr_features_df.rename(
            columns={curr_features.ids.name: old_features.ids.name})
        old_user_features_df = old_features.df
        curr_features_df = curr_features_df[old_user_features_df.columns]
        updated_user_features_df = pd.concat([old_user_features_df, curr_features_df], sort=False)
        duplicated_ids = updated_user_features_df.duplicated(subset=old_features.ids.name)
        updated_user_features_df = updated_user_features_df[~duplicated_ids]

        return FeatureDataset(updated_user_features_df, name=curr_features.name)
