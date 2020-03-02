import tensorflow as tf
from azureml.studio.core.logger import module_logger


class TrainLogHook(tf.estimator.SessionRunHook):
    def __init__(self, steps_per_iter):
        if steps_per_iter == 0:
            raise ValueError(f"Steps in every iteration must great than zero.")
        self.steps_per_iter = steps_per_iter
        self.step = 0

    def begin(self):
        self.step = 0

    def after_run(self, run_context, run_values):
        self.step += 1
        if self.step % self.steps_per_iter == 0:
            module_logger.info(f"Iteration {int(self.step / self.steps_per_iter)} finished.")
