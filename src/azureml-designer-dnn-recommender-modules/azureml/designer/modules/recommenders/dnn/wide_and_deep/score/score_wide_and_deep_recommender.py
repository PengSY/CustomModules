from azureml.studio.core.logger import module_logger
from azureml.designer.modules.recommenders.dnn.common.extend_types import RecommenderPredictionKind, \
    RecommendedItemSelection, \
    Boolean
from azureml.designer.modules.recommenders.dnn.common.utils import before_init, before_run
from azureml.designer.modules.recommenders.dnn.common.dataset import InteractionDataset, FeatureDataset
from azureml.designer.modules.recommenders.dnn.common.wide_and_deep_model import WideAndDeepModel
from azureml.designer.modules.recommenders.dnn.wide_and_deep.score. \
    rating_prediction_scorer import RatingPredictionScorer
from azureml.designer.modules.recommenders.dnn.wide_and_deep.score. \
    recommend_rated_item_scorer import RecommendRatedItemScorer
from azureml.designer.modules.recommenders.dnn.wide_and_deep.score. \
    recommend_unrated_item_scorer import RecommendUnratedItemScorer
from azureml.designer.modules.recommenders.dnn.wide_and_deep.score.recommend_all_items_scorer import \
    RecommendAllItemScorer


class ScoreWideAndDeepRecommenderModule:
    @before_init
    def __init__(self,
                 prediction_kind: RecommenderPredictionKind = None,
                 recommended_item_selection: RecommendedItemSelection = None,
                 max_recommended_item_count: int = None,
                 min_recommendation_pool_size: int = None,
                 return_ratings: Boolean = None):
        module_logger.info(f"Init score params.")
        self.prediction_kind = prediction_kind
        self.recommended_item_selection = recommended_item_selection
        self.max_recommended_item_count = max_recommended_item_count
        self.min_recommendation_pool_size = min_recommendation_pool_size
        self.return_ratings = return_ratings

    @staticmethod
    def get_scorer(prediction_kind: RecommenderPredictionKind, recommended_item_selection: RecommendedItemSelection):
        if prediction_kind == RecommenderPredictionKind.RatingPrediction:
            return RatingPredictionScorer()
        elif prediction_kind == RecommenderPredictionKind.ItemRecommendation:
            if recommended_item_selection == RecommendedItemSelection.FromAllItems:
                return RecommendAllItemScorer()
            elif recommended_item_selection == RecommendedItemSelection.FromRatedItems:
                return RecommendRatedItemScorer()
            elif recommended_item_selection == RecommendedItemSelection.FromUnratedItems:
                return RecommendUnratedItemScorer()
            else:
                raise NotImplementedError(f"{recommended_item_selection} not supported now.")
        else:
            raise NotImplementedError(f"{prediction_kind} and {recommended_item_selection} not supported now.")

    def update_params(self,
                      prediction_kind: RecommenderPredictionKind = None,
                      recommended_item_selection: RecommendedItemSelection = None,
                      max_recommended_item_count: int = None,
                      min_recommendation_pool_size: int = None,
                      return_ratings: Boolean = None):
        for attr_name, attr_value in locals().items():
            if attr_name != "self" and attr_value is not None:
                setattr(self, attr_name, attr_value)

    @before_run
    def run(self,
            learner: WideAndDeepModel,
            test_interactions: InteractionDataset,
            training_interactions: InteractionDataset,
            user_features: FeatureDataset,
            item_features: FeatureDataset,
            prediction_kind: RecommenderPredictionKind = None,
            recommended_item_selection: RecommendedItemSelection = None,
            max_recommended_item_count: int = None,
            min_recommendation_pool_size: int = None,
            return_ratings: Boolean = None):
        module_logger.info(f"Update score params.")
        self.update_params(prediction_kind, recommended_item_selection, max_recommended_item_count,
                           min_recommendation_pool_size, return_ratings)
        module_logger.info(f"Get scorer.")
        scorer = self.get_scorer(self.prediction_kind, self.recommended_item_selection)
        res = scorer.score(learner,
                           test_interactions=test_interactions,
                           user_features=user_features,
                           item_features=item_features,
                           training_interactions=training_interactions,
                           max_recommended_item_count=self.max_recommended_item_count,
                           min_recommendation_pool_size=self.min_recommendation_pool_size,
                           return_ratings=self.return_ratings)
        return res
