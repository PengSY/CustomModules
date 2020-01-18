import itertools
import math
import tensorflow as tf
import pandas as pd
from azureml.studio.core.logger import module_logger
from azureml.studio.core.error import UserError
from azureml.designer.modules.recommenders.dnn.common.feature_builder import FeatureBuilder
from azureml.designer.modules.recommenders.dnn.common.constants import RANDOM_SEED, USER_INTERNAL_KEY, ITEM_INTERNAL_KEY
from azureml.designer.modules.recommenders.dnn.common.extend_types import DeepActivationSelection, OptimizerSelection
from azureml.designer.modules.recommenders.dnn.common.dataset import WideDeepDataset, FeatureDataset, InteractionDataset
from azureml.designer.modules.recommenders.dnn.wide_and_deep.train. \
    train_log_hook import TrainLogHook
import horovod.tensorflow as hvd


class NanLossDuringTrainingError(UserError):
    def __init__(self):
        msg = "Training stopped with NanLossDuringTrainingError. " \
              "Please try other optimizers, smaller batch size and/or smaller learning rate."
        super().__init__(msg)


class WideAndDeepModel:
    OPTIMIZERS = {OptimizerSelection.Adagrad: tf.optimizers.Adagrad,
                  OptimizerSelection.Adam: tf.optimizers.Adam,
                  OptimizerSelection.Ftrl: tf.optimizers.Ftrl,
                  OptimizerSelection.RMSProp: tf.optimizers.RMSprop,
                  OptimizerSelection.SGD: tf.optimizers.SGD,
                  OptimizerSelection.Adadelta: tf.keras.optimizers.Adadelta}
    ACTIVATION_FNS = {DeepActivationSelection.ReLU: tf.nn.relu,
                      DeepActivationSelection.Sigmoid: tf.keras.activations.sigmoid,
                      DeepActivationSelection.Tanh: tf.keras.activations.tanh,
                      DeepActivationSelection.Linear: tf.keras.activations.linear,
                      DeepActivationSelection.LeakyReLU: tf.nn.leaky_relu}

    def __init__(self, epochs, batch_size, wide_part_optimizer, wide_learning_rate, deep_part_optimizer,
                 deep_learning_rate, deep_hidden_units, deep_activation_fn, deep_dropout, batch_norm, crossed_dim,
                 user_dim, item_dim, categorical_feature_dim, model_dir):
        self.epochs = epochs
        self.batch_size = batch_size
        self.wide_part_optimizer = wide_part_optimizer
        self.wide_learning_rate = wide_learning_rate
        self.deep_part_optimizer = deep_part_optimizer
        self.deep_learning_rate = deep_learning_rate
        self.deep_hidden_units = deep_hidden_units
        self.deep_activation_fn = deep_activation_fn
        self.deep_dropout = deep_dropout
        self.batch_norm = batch_norm
        self.crossed_dim = crossed_dim
        self.user_dim = user_dim
        self.item_dim = item_dim
        self.feature_dim = categorical_feature_dim
        self.model_dir = model_dir

        self.steps_per_iteration = None
        self.feature_builder = None

        self.user_features = None
        self.item_features = None

        hvd.init()
        self.hvd_rank = hvd.rank()
        tf.config.experimental.set_visible_devices([str(hvd.local_rank())], 'GPU')
        if self.hvd_rank != 0:
            self.model_dir = None

    def _build_column_feature(self):
        users = tf.feature_column.categorical_column_with_vocabulary_list(
            key=USER_INTERNAL_KEY,
            vocabulary_list=self.feature_builder.user_vocab,
            num_oov_buckets=1)
        items = tf.feature_column.categorical_column_with_vocabulary_list(
            key=ITEM_INTERNAL_KEY,
            vocabulary_list=self.feature_builder.item_vocab,
            num_oov_buckets=1)
        wide_columns = self._build_wide_feature_columns(users, items)
        deep_columns = self._build_deep_feature_columns(users, items)
        return wide_columns, deep_columns

    def _build_wide_feature_columns(self, users, items):
        crossed_feature = tf.feature_column.crossed_column(keys=[users, items], hash_bucket_size=self.crossed_dim)
        return [users, items, crossed_feature]

    def _build_deep_feature_columns(self, users, items):
        deep_columns = []
        deep_columns.append(tf.feature_column.embedding_column(categorical_column=users, dimension=self.user_dim,
                                                               max_norm=self.user_dim ** 0.5))
        deep_columns.append(tf.feature_column.embedding_column(categorical_column=items, dimension=self.item_dim,
                                                               max_norm=self.item_dim ** 0.5))
        for feature_meta in self.feature_builder.user_feature_metas + self.feature_builder.item_feature_metas:
            if not feature_meta.is_categorical():
                deep_columns.append(
                    tf.feature_column.numeric_column(key=feature_meta.internal_name, shape=feature_meta.shape))
            else:
                categorical_feature = tf.feature_column.categorical_column_with_vocabulary_list(
                    key=feature_meta.internal_name, vocabulary_list=feature_meta.vocab, num_oov_buckets=1)
                deep_columns.append(
                    tf.feature_column.embedding_column(categorical_column=categorical_feature,
                                                       dimension=self.feature_dim,
                                                       max_norm=self.feature_dim ** 0.5))

        return deep_columns

    def _build_optimizer(self, optimizer_name, learning_rate):
        try:
            optimizer = self.OPTIMIZERS[optimizer_name]
        except KeyError:
            raise ValueError(f"Unsupported optimizer {optimizer_name}")
        distributed_optimizer = hvd.DistributedOptimizer(optimizer(learning_rate=learning_rate * hvd.size()))
        return distributed_optimizer

    def _build_activation_fn(self, activation_fn_name):
        try:
            activation_fn = self.ACTIVATION_FNS[activation_fn_name]
        except KeyError:
            raise ValueError(f"Unsupported activation function {activation_fn_name}")
        return activation_fn

    def _build_model(self):
        module_logger.info(f"Build model.")
        config = tf.estimator.RunConfig(tf_random_seed=RANDOM_SEED,
                                        save_summary_steps=None,
                                        keep_checkpoint_max=1,
                                        log_step_count_steps=self.steps_per_iteration,
                                        save_checkpoints_steps=self.steps_per_iteration)
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
        wide_columns, deep_columns = self._build_column_feature()
        wide_optimizer = self._build_optimizer(self.wide_part_optimizer, learning_rate=self.wide_learning_rate)
        deep_optimizer = self._build_optimizer(self.deep_part_optimizer, learning_rate=self.deep_learning_rate)
        deep_activation_fn = self._build_activation_fn(activation_fn_name=self.deep_activation_fn)
        model = tf.estimator.DNNLinearCombinedRegressor(model_dir=self.model_dir,
                                                        linear_feature_columns=wide_columns,
                                                        linear_optimizer=wide_optimizer,
                                                        dnn_feature_columns=deep_columns,
                                                        dnn_optimizer=deep_optimizer,
                                                        dnn_hidden_units=self.deep_hidden_units,
                                                        dnn_activation_fn=deep_activation_fn,
                                                        dnn_dropout=self.deep_dropout,
                                                        config=config,
                                                        batch_norm=self.batch_norm)
        module_logger.info(f"Build model:\n"
                           f"Epochs: {self.epochs}\n"
                           f"Batch size: {self.batch_size}\n"
                           f"Wide part optimizer: {self.wide_part_optimizer}\n"
                           f"Wide learning rate: {self.wide_learning_rate}\n"
                           f"Crossed dimension: {self.crossed_dim}\n"
                           f"Deep part optimizer: {self.deep_part_optimizer}\n"
                           f"Deep learning rate: {self.deep_learning_rate}\n"
                           f"User dimension: {self.user_dim}\n"
                           f"Item dimension: {self.item_dim}\n"
                           f"Categorical feature dimension: {self.item_dim}\n"
                           f"Hidden units: {self.deep_hidden_units}\n"
                           f"Activation function: {self.deep_activation_fn}\n"
                           f"Dropout: {self.deep_dropout}\n"
                           f"Batch norm: {self.batch_norm}\n")
        return model

    def train(self, interactions: InteractionDataset, user_features: FeatureDataset = None,
              item_features: FeatureDataset = None):
        training_data = WideDeepDataset(interactions=interactions, user_features=user_features,
                                        item_features=item_features)
        module_logger.info(f"Fit and build features.")
        self.feature_builder = FeatureBuilder().fit(dataset=training_data, user_features=user_features,
                                                    item_features=item_features)
        training_data = self.feature_builder.build(training_data)
        self.steps_per_iteration = math.ceil(training_data.row_size / self.batch_size)
        model = self._build_model()
        input_fn = training_data.get_input_handler(batch_size=self.batch_size, epochs=self.epochs / hvd.size(),
                                                   shuffle=True)
        log_hook = TrainLogHook(steps_per_iter=self.steps_per_iteration)
        module_logger.info(f"Start to train model, rank {self.hvd_rank}")
        bcast_hook = hvd.BroadcastGlobalVariablesHook(0)
        try:
            model.train(input_fn=input_fn, hooks=[log_hook, bcast_hook])
        except tf.estimator.NanLossDuringTrainingError:
            raise NanLossDuringTrainingError

    def predict(self, interactions: InteractionDataset, user_features: FeatureDataset = None,
                item_features: FeatureDataset = None):
        module_logger.info(f"Update features.")
        self.feature_builder = self.feature_builder.update_features(user_features=user_features,
                                                                    item_features=item_features)
        test_data = WideDeepDataset(interactions=interactions, user_features=self.feature_builder.user_features,
                                    item_features=self.feature_builder.item_features)
        module_logger.info(f"Build features.")
        test_data = self.feature_builder.build(test_data)
        model = self._build_model()
        input_fn = test_data.get_input_handler(batch_size=self.batch_size)
        module_logger.info(f"Generate predictions.")
        predictions = list(itertools.islice(model.predict(input_fn=input_fn), test_data.row_size))
        predictions = pd.Series([p["predictions"][0] for p in predictions])
        return predictions
