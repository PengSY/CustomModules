from azureml.designer.modules.recommenders.dnn.wide_and_deep.train.train_wide_and_deep_recommender import \
    TrainWideAndDeepRecommenderModule
from azureml.designer.modules.recommenders.dnn.common.entry_utils import build_cli_args


def main():
    kwargs = build_cli_args(TrainWideAndDeepRecommenderModule.run)
    TrainWideAndDeepRecommenderModule().run(**kwargs)


if __name__ == "__main__":
    main()
