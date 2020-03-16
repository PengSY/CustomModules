import pandas as pd
import numpy as np
from azureml.studio.internal.error import ErrorMapping, ColumnNotFoundError
from azureml.designer.modules.recommenders.dnn.common.dataset import FeatureDataset
from azureml.studio.core.data_frame_schema import ColumnTypeName
from azureml.designer.modules.recommenders.dnn.common.utils import convert_to_str


class FeatureMeta:
    def __init__(self, name, type_, fill, vocab=None):
        self.name = name
        self.type_ = type_
        self.fill = fill
        self.vocab = vocab

    def is_numeric_feature(self):
        return self.vocab is None


class FeatureBuilder:
    def __init__(self, ids: pd.Series, id_key: str, features: FeatureDataset = None, feat_key_suffix=None):
        ids = convert_to_str(ids).dropna().unique()
        self.id_vocab = pd.Series(ids, name=id_key)

        self._features_df = None
        self._feature_metas = {}
        if features is not None:
            features_df = pd.DataFrame({id_key: convert_to_str(features.ids)})
            for idx, name in enumerate(features.features.columns):
                feature = features.features[name]
                column_type = features.get_column_type(name)
                key = f"{name}_{feat_key_suffix}_{idx}"

                normalized_feature = self._normalize_feature(feature=feature, column_type=column_type)
                feature_meta = self._init_feature_meta(feature=normalized_feature, column_type=column_type)
                if feature_meta is not None:
                    features_df[key] = normalized_feature
                    self.feature_metas[key] = feature_meta

            if self.feature_metas:
                self._features_df = features_df

    def build(self, ids: pd.Series):
        # should ensure returned ids are equal order and length with the input
        ids = convert_to_str(ids)
        features_df = pd.DataFrame({self.id_key: ids})

        if self.feature_metas:
            features_df = pd.merge(features_df, self._features_df, how='left', on=[self.id_key])
            for key, meta in self.feature_metas.items():
                feature = features_df[key]
                features_df[key] = self._fill_feature_na(feature=feature, feature_meta=meta)

        return features_df

    def update(self, features: FeatureDataset):
        if not self.feature_metas or not features:
            return

        self._check_features(features)
        feature_names = [meta.name for key, meta in self.feature_metas.items()]
        new_features_df = features.df[[features.ids.name, *feature_names]]
        new_features_df = new_features_df.rename(
            columns=dict((meta.name, key) for key, meta in self.feature_metas.items()))
        new_features_df = new_features_df.rename(columns={features.ids.name: self.id_key})
        new_features_df[self.id_key] = convert_to_str(new_features_df[self.id_key])
        new_features_df = pd.concat([self._features_df, new_features_df], axis=0)
        existed_ids = new_features_df.duplicated(subset=self.id_key, keep='first')
        new_features_df = new_features_df[~existed_ids]

        self._features_df = new_features_df

    def _check_features(self, features: FeatureDataset):
        for _, feature_meta in self.feature_metas.items():
            name = feature_meta.name
            if name not in features.features:
                ErrorMapping.throw(ColumnNotFoundError(column_id=name, arg_name_missing_column=features.name))
            column_type = features.get_column_type(name)
            if features.get_column_type(name) != feature_meta.type_:
                ErrorMapping.verify_element_type(type_=column_type, expected_type=feature_meta.type_, column_name=name,
                                                 arg_name=features.name)

    @staticmethod
    def _fill_feature_na(feature: pd.Series, feature_meta: FeatureMeta):
        if feature_meta.type_ == ColumnTypeName.NUMERIC:
            feature = feature.replace(to_replace=[np.inf, -np.inf], value=np.nan)
        feature = feature.fillna(feature_meta.fill)

        return feature

    @staticmethod
    def _normalize_feature(feature: pd.Series, column_type):
        if column_type in [ColumnTypeName.CATEGORICAL, ColumnTypeName.BINARY, ColumnTypeName.OBJECT]:
            new_feature = convert_to_str(column=feature)
        else:
            new_feature = feature.copy()

        return new_feature

    @staticmethod
    def _init_feature_meta(feature: pd.Series, column_type):
        if column_type == ColumnTypeName.DATETIME or column_type == ColumnTypeName.TIMESPAN:
            feature_meta = None
        elif column_type in [ColumnTypeName.CATEGORICAL, ColumnTypeName.BINARY, ColumnTypeName.OBJECT,
                             ColumnTypeName.STRING]:
            feature_meta = FeatureMeta(name=feature.name, type_=column_type, fill='', vocab=feature.dropna().unique())
        else:
            mean = feature.replace(to_replace=[np.inf, -np.inf], value=np.nan).mean()
            feature_meta = FeatureMeta(name=feature.name, type_=column_type, fill=mean)

        return feature_meta

    @property
    def feature_metas(self):
        return self._feature_metas

    @property
    def id_key(self):
        return self.id_vocab.name
