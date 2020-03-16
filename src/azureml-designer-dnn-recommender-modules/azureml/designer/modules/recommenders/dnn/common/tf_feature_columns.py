from abc import abstractmethod
import tensorflow as tf


class FeatureColumn:
    def __init__(self):
        self._feature_column = None

    @abstractmethod
    def build(self):
        pass

    def reset(self):
        self._feature_column = None

    @property
    def feature_column(self):
        return self._feature_column


class CategoricalVocabListFeatureColumn(FeatureColumn):
    def __init__(self, key: str, vocab):
        super().__init__()
        self.key = key
        self.vocab = vocab

    def build(self):
        if not self._feature_column:
            self._feature_column = tf.feature_column.categorical_column_with_vocabulary_list(
                key=self.key,
                vocabulary_list=self.vocab,
                num_oov_buckets=1
            )

        return self.feature_column


class NumericFeatureColumn(FeatureColumn):
    def __init__(self, key: str, shape=(1,)):
        super().__init__()
        self.key = key
        self.shape = shape

    def build(self):
        if not self.feature_column:
            self._feature_column = tf.feature_column.numeric_column(key=self.key, shape=self.shape)

        return self.feature_column


class CrossedFeatureColumn(FeatureColumn):
    def __init__(self, categorical_features, hash_bucket_size):
        super().__init__()
        self.categorical_features = categorical_features
        self.hash_bucket_size = hash_bucket_size

    def build(self):
        if not self.feature_column:
            keys = [feature.build() for feature in self.categorical_features]
            self._feature_column = tf.feature_column.crossed_column(keys=keys, hash_bucket_size=self.hash_bucket_size)

        return self.feature_column


class EmbeddingFeatureColumn(FeatureColumn):
    def __init__(self, categorical_feature, dimension, max_norm=None):
        super().__init__()
        self.categorical_feature = categorical_feature
        self.dimension = dimension
        self.max_norm = max_norm if max_norm else dimension ** 0.5

    def build(self):
        if not self.feature_column:
            self._feature_column = tf.feature_column.embedding_column(
                categorical_column=self.categorical_feature.build(),
                dimension=self.dimension,
                max_norm=self.max_norm)

        return self.feature_column


def is_basic_feature(feature):
    if not isinstance(feature, FeatureColumn):
        raise TypeError(f"feature should be of {FeatureColumn.__name__} type")
    return isinstance(feature, (CategoricalVocabListFeatureColumn, NumericFeatureColumn))


def parse_basic_features(feature_columns):
    basic_features = set()
    for feature in feature_columns:
        if is_basic_feature(feature):
            basic_features.add(feature)
        elif isinstance(feature, CrossedFeatureColumn):
            for basic_feature in feature.categorical_features:
                basic_features.add(basic_feature)
        elif isinstance(feature, EmbeddingFeatureColumn):
            basic_features.add(feature.categorical_feature)
        else:
            raise NotImplementedError

    return list(basic_features)
