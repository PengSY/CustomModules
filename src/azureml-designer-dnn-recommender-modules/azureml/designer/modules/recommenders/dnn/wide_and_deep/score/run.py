import os
from azureml.studio.core.io.data_frame_directory import save_data_frame_to_directory
from azureml.studio.core.data_frame_schema import DataFrameSchema
from azureml.designer.modules.recommenders.dnn.common.utils import get_spec_dir
from azureml.designer.modules.recommenders.dnn.common.arg_parser import parse_command_line_args
from azureml.designer.modules.recommenders.dnn.wide_and_deep.score. \
    score_wide_and_deep_recommender import ScoreWideAndDeepRecommenderModule


def main():
    input_normal_params, input_port_params, output_params = parse_command_line_args(
        module_spec=os.path.join(get_spec_dir(), "score_wide_and_deep_recommender.yaml"))
    res = ScoreWideAndDeepRecommenderModule(**input_normal_params).run(**input_port_params)
    save_data_frame_to_directory(save_to=output_params['scored_data'], data=res,
                                 schema=DataFrameSchema.data_frame_to_dict(res))


if __name__ == "__main__":
    main()
