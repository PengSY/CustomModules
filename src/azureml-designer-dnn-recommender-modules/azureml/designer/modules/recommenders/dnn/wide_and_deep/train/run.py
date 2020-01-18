import sys
import os
import shutil

# def set_environment():
#     src_folder = os.path.realpath(
#         os.path.join(
#             os.path.abspath(__file__),
#             "../../../../../../../../../azureml-designer-dnn-recommender-modules",
#         )
#     )
#     sys.path.insert(0, src_folder)
#     import azureml
#     print(azureml.__path__)
#     designer_folder = os.path.join(azureml.__path__[0], 'designer')
#     if not os.path.exists(designer_folder):
#         src = os.path.join(src_folder, 'azureml', 'designer')
#         dst = designer_folder
#         print(src, dst)
#         shutil.copytree(src, dst)
#
#
# set_environment()

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
