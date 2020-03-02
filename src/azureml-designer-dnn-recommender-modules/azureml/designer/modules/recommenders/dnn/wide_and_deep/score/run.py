from azureml.designer.modules.recommenders.dnn.wide_and_deep.score. \
    score_wide_and_deep_recommender import ScoreWideAndDeepRecommenderModule
from azureml.designer.modules.recommenders.dnn.common.entry_utils import build_cli_args


def main():
    kwargs = build_cli_args(ScoreWideAndDeepRecommenderModule.run)
    ScoreWideAndDeepRecommenderModule().run(**kwargs)


if __name__ == "__main__":
    main()
