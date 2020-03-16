import os
import pickle
import datetime
import pandas as pd
import tensorflow as tf
import numpy as np
from enum import Enum
from time import time
from azureml.studio.core.logger import module_logger
from azureml.designer.modules.recommenders.dnn.common.entry_param import EntryParam
from azureml.studio.core.error import UserError
from azureml.studio.internal.error import InvalidLearnerError
from azureml.designer.modules.recommenders.dnn.common.feature_builder import FeatureBuilder
from azureml.designer.modules.recommenders.dnn.common.tf_feature_columns import CategoricalVocabListFeatureColumn, \
    NumericFeatureColumn, CrossedFeatureColumn, EmbeddingFeatureColumn, parse_basic_features
from azureml.designer.modules.recommenders.dnn.common.constants import RANDOM_SEED, MODEL_SAVE_FILE
from azureml.designer.modules.recommenders.dnn.common.dataset import TransactionDataset, FeatureDataset


class ActivationFnSelection(Enum):
    ReLU = 'ReLU'
    Sigmoid = 'Sigmoid'
    Tanh = 'Tanh'
    Linear = 'Linear'
    LeakyReLU = 'LeakyReLU'


class OptimizerSelection(Enum):
    Adagrad = 'Adagrad'
    Adam = 'Adam'
    Ftrl = 'Ftrl'
    RMSProp = 'RMSProp'
    SGD = 'SGD'
    Adadelta = 'Adadelta'


class NanLossDuringTrainingError(UserError):
    def __init__(self):
        msg = "Training stopped with NanLossDuringTrainingError. " \
              "Please try other optimizers, smaller batch size and/or smaller learning rate."
        super().__init__(msg)


class WideNDeepModelHyperParams:
    def __init__(self,
                 epochs,
                 batch_size,
                 wide_optimizer,
                 wide_lr,
                 deep_optimizer,
                 deep_lr,
                 hidden_units,
                 activation_fn,
                 dropout,
                 batch_norm,
                 crossed_dim,
                 user_dim,
                 item_dim,
                 embed_dim):
        self.epochs = epochs
        self.batch_size = batch_size
        self.wide_optimizer = wide_optimizer
        self.wide_lr = wide_lr
        self.deep_optimizer = deep_optimizer
        self.deep_lr = deep_lr
        self.hidden_units = hidden_units
        self.activation_fn = activation_fn
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.crossed_dim = crossed_dim
        self.user_dim = user_dim
        self.item_dim = item_dim
        self.embed_dim = embed_dim

    def __str__(self):
        hyper_param_info = f"Epochs: {self.epochs}\n" \
            f"Batch size: {self.batch_size}\n" \
            f"Wide optimizer: {self.wide_optimizer}\n" \
            f"Wide learning rate: {self.wide_lr}\n" \
            f"Deep optimizer: {self.deep_optimizer}\n" \
            f"Deep learning rate: {self.deep_lr}\n" \
            f"Hidden units: {self.hidden_units}\n" \
            f"Activation function: {self.activation_fn}\n" \
            f"Dropout: {self.dropout}\n" \
            f"Batch norm: {self.batch_norm}\n" \
            f"Crossed dimension: {self.crossed_dim}\n" \
            f"User embedding dimension: {self.user_dim}\n" \
            f"Item embedding dimension: {self.item_dim}\n" \
            f"Categorical feature embedding dimension: {self.embed_dim}"

        return hyper_param_info


class WideNDeepModel(metaclass=EntryParam):
    MODEL_NAME = "Wide & Deep Recommendation Model"

    def __init__(self, hyper_params: WideNDeepModelHyperParams, save_dir, user_feature_builder: FeatureBuilder,
                 item_feature_builder: FeatureBuilder, wide_columns=None, deep_columns=None, random_seed=RANDOM_SEED,
                 rel_checkpoints_dir='checkpoints'):
        self.hyper_params = hyper_params
        self.save_dir = save_dir
        self.random_seed = random_seed
        self.rel_checkpoints_dir = rel_checkpoints_dir
        self.user_feature_builder = user_feature_builder
        self.item_feature_builder = item_feature_builder
        self.wide_columns = wide_columns
        self.deep_columns = deep_columns
        self.estimator = None

    def default_columns(self):
        user = CategoricalVocabListFeatureColumn(key=self.user_feature_builder.id_key,
                                                 vocab=self.user_feature_builder.id_vocab)
        item = CategoricalVocabListFeatureColumn(key=self.item_feature_builder.id_key,
                                                 vocab=self.item_feature_builder.id_vocab)

        # wide feature columns
        crossed_feature = CrossedFeatureColumn(categorical_features=[user, item],
                                               hash_bucket_size=self.hyper_params.crossed_dim)
        wide_columns = [user, item, crossed_feature]

        # deep feature columns
        user_embed = EmbeddingFeatureColumn(categorical_feature=user, dimension=self.hyper_params.user_dim)
        item_embed = EmbeddingFeatureColumn(categorical_feature=item, dimension=self.hyper_params.item_dim)
        deep_columns = [user_embed, item_embed]

        for key, feature_meta in {**self.user_feature_builder.feature_metas,
                                  **self.item_feature_builder.feature_metas}.items():
            if feature_meta.is_numeric_feature():
                deep_columns.append(NumericFeatureColumn(key=key))
            else:
                cat_feature = CategoricalVocabListFeatureColumn(key=key, vocab=feature_meta.vocab)
                deep_columns.append(
                    EmbeddingFeatureColumn(categorical_feature=cat_feature, dimension=self.hyper_params.embed_dim))

        return wide_columns, deep_columns

    def build_model(self, run_config=None):
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
        if run_config is None:
            run_config = tf.estimator.RunConfig(tf_random_seed=self.random_seed,  # fix random sedd
                                                save_summary_steps=None,  # disable summary
                                                save_checkpoints_secs=None,  # disable checkpoints
                                                save_checkpoints_steps=None)  # disable checkpoints
        # build feature columns
        wide_columns, deep_columns = self._build_feature_columns()
        # build optimizers
        wide_optimizer = self.build_optimizer(optimizer_name=self.hyper_params.wide_optimizer,
                                              learning_rate=self.hyper_params.wide_lr)
        deep_optimizer = self.build_optimizer(optimizer_name=self.hyper_params.deep_optimizer,
                                              learning_rate=self.hyper_params.deep_lr)
        # build activation function
        activation_fn = self.build_activation_fn(activation_fn_name=self.hyper_params.activation_fn)

        self.estimator = tf.estimator.DNNLinearCombinedRegressor(linear_feature_columns=wide_columns,
                                                                 linear_optimizer=wide_optimizer,
                                                                 dnn_feature_columns=deep_columns,
                                                                 dnn_optimizer=deep_optimizer,
                                                                 dnn_hidden_units=self.hyper_params.hidden_units,
                                                                 dnn_activation_fn=activation_fn,
                                                                 dnn_dropout=self.hyper_params.dropout,
                                                                 batch_norm=self.hyper_params.batch_norm,
                                                                 config=run_config,
                                                                 model_dir=self.checkpoints_dir)

    def train(self, transactions: TransactionDataset):
        instances_count = transactions.row_size
        batches_count = np.ceil(instances_count / self.hyper_params.batch_size)
        module_logger.info(f"Get {instances_count} training instances, and {batches_count} batches per epoch.")
        run_config = tf.estimator.RunConfig(tf_random_seed=self.random_seed,
                                            log_step_count_steps=batches_count,  # log loss after each epoch
                                            save_checkpoints_steps=batches_count * self.hyper_params.epochs,
                                            keep_checkpoint_max=1)
        module_logger.info(f"Build model:\n{self.hyper_params}")
        self.build_model(run_config=run_config)
        input_fn = self.get_input_fn(transactions=transactions, batch_size=self.hyper_params.batch_size,
                                     epochs=self.hyper_params.epochs, shuffle=True)
        try:
            self.estimator.train(input_fn=input_fn)
        except tf.estimator.NanLossDuringTrainingError:
            raise NanLossDuringTrainingError

    def predict(self, transactions: TransactionDataset, log_every_n_instances: int = 100000):
        if transactions.row_size == 0:
            return pd.Series()

        instances_count = transactions.row_size
        module_logger.info(f"Get {instances_count} test instances")
        module_logger.info(f"Rebuild model:\n {self.hyper_params}")
        self.build_model()
        input_fn = self.get_input_fn(transactions=transactions, batch_size=self.hyper_params.batch_size)
        predictions = []
        start_time = time()
        for p in self.estimator.predict(input_fn=input_fn):
            if len(predictions) % log_every_n_instances == 0 and len(predictions) > 0:
                cost_seconds = time() - start_time
                remaining_seconds = cost_seconds / len(predictions) * (instances_count - len(predictions))
                module_logger.info(f"Finished {len(predictions)} instance predictions, "
                                   f"cost time: {datetime.timedelta(seconds=cost_seconds)}."
                                   f"Remaining time: {datetime.timedelta(seconds=remaining_seconds)}")
            predictions.append(p["predictions"][0])
        module_logger.info(f"Finished {len(predictions)} instance predictions. "
                           f"Cost time: {datetime.timedelta(seconds=(time() - start_time))}")

        predictions = pd.Series(predictions)

        return predictions

    def update_feature_builders(self, user_features: FeatureDataset, item_features: FeatureDataset):
        self.user_feature_builder.update(features=user_features)
        self.item_feature_builder.update(features=item_features)

    @staticmethod
    def build_optimizer(optimizer_name: OptimizerSelection, learning_rate):
        optimizers = {OptimizerSelection.Adagrad: tf.optimizers.Adagrad,
                      OptimizerSelection.Adam: tf.optimizers.Adam,
                      OptimizerSelection.Ftrl: tf.optimizers.Ftrl,
                      OptimizerSelection.RMSProp: tf.optimizers.RMSprop,
                      OptimizerSelection.SGD: tf.optimizers.SGD,
                      OptimizerSelection.Adadelta: tf.keras.optimizers.Adadelta}
        try:
            optimizer = optimizers[optimizer_name]
        except KeyError:
            raise ValueError(f"Unsupported optimizer {optimizer_name}")

        return optimizer(learning_rate=learning_rate)

    @staticmethod
    def build_activation_fn(activation_fn_name: ActivationFnSelection):
        activation_fns = {ActivationFnSelection.ReLU: tf.nn.relu,
                          ActivationFnSelection.Sigmoid: tf.keras.activations.sigmoid,
                          ActivationFnSelection.Tanh: tf.keras.activations.tanh,
                          ActivationFnSelection.Linear: tf.keras.activations.linear,
                          ActivationFnSelection.LeakyReLU: tf.nn.leaky_relu}
        try:
            activation_fn = activation_fns[activation_fn_name]
        except KeyError:
            raise ValueError(f"Unsupported activation function {activation_fn_name}")

        return activation_fn

    def get_input_fn(self, transactions: TransactionDataset, batch_size, epochs=1, shuffle=False):
        user_ids = transactions.users
        item_ids = transactions.items
        user_features_df = self.user_feature_builder.build(ids=user_ids)
        item_features_df = self.item_feature_builder.build(ids=item_ids)

        x_df = pd.concat([user_features_df, item_features_df], axis=1)
        y_sr = transactions.ratings

        return tf.compat.v1.estimator.inputs.pandas_input_fn(x=x_df, y=y_sr, batch_size=batch_size, num_epochs=epochs,
                                                             shuffle=shuffle)

    def _build_feature_columns(self):
        # if not specify wide and deep feature columns, use default scheme
        if self.wide_columns is None and self.deep_columns is None:
            self.wide_columns, self.deep_columns = self.default_columns()
        # check if basic feature columns are in feature builders
        self._check_feature_columns()
        wide_columns = [feature_column.build() for feature_column in self.wide_columns]
        deep_columns = [feature_column.build() for feature_column in self.deep_columns]

        return wide_columns, deep_columns

    def _check_feature_columns(self):
        basic_features = parse_basic_features(feature_columns=[*self.wide_columns, *self.deep_columns])
        module_logger.info(f"Model is expected to be fed with features: {[f.key for f in basic_features]}")
        feature_keys = {*self.user_feature_builder.feature_metas.keys(),
                        *self.item_feature_builder.feature_metas.keys(),
                        self.user_feature_builder.id_key,
                        self.item_feature_builder.id_key}
        for feature in basic_features:
            if feature.key not in feature_keys:
                raise RuntimeError(f"feature {feature.key} not found in feature datasets.")

    def save(self):
        self.estimator = None
        for feature_column in [*self.wide_columns, *self.deep_columns]:
            feature_column.reset()
        save_path = os.path.join(self.save_dir, MODEL_SAVE_FILE)
        with open(save_path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, save_dir: str):
        try:
            with open(os.path.join(save_dir, MODEL_SAVE_FILE), "rb") as f:
                model = pickle.load(f)
        except ModuleNotFoundError:
            raise InvalidLearnerError(arg_name=cls.MODEL_NAME)
        if isinstance(model, cls):
            model.save_dir = save_dir
        else:
            raise InvalidLearnerError(arg_name=cls.MODEL_NAME)
        return model

    @property
    def checkpoints_dir(self):
        return os.path.join(self.save_dir, self.rel_checkpoints_dir)
