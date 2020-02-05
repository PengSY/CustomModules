import os
from azureml.designer.modules.recommenders.dnn.common.utils import get_spec_dir
from azureml.designer.modules.recommenders.dnn.common.arg_parser import parse_command_line_args
from azureml.designer.modules.recommenders.dnn.wide_and_deep.train.train_wide_and_deep_recommender import \
    TrainWideAndDeepRecommenderModule


def main():
    input_port_params, input_normal_params, output_params = parse_command_line_args(
        module_spec=os.path.join(get_spec_dir(), "train_wide_and_deep_recommender.yaml"))
    TrainWideAndDeepRecommenderModule().run(**input_port_params, **input_normal_params, **output_params)


if __name__ == "__main__":
    main()
