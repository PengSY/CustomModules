from enum import Enum
from azureml.studio.core.data_frame_schema import DataFrameSchema
from azureml.studio.core.io.data_frame_directory import save_data_frame_to_directory
from azureml.studio.core.logger import module_logger
from azureml.studio.internal.error import ErrorMapping
from azureml.designer.modules.recommenders.dnn.common.dataset import InteractionDataset, FeatureDataset
from azureml.designer.modules.recommenders.dnn.wide_and_deep.common.wide_and_deep_model import WideAndDeepModel
from azureml.designer.modules.recommenders.dnn.wide_and_deep.score. \
    rating_prediction_scorer import RatingPredictionScorer
from azureml.designer.modules.recommenders.dnn.wide_and_deep.score. \
    recommend_rated_item_scorer import RecommendRatedItemScorer
from azureml.designer.modules.recommenders.dnn.wide_and_deep.score. \
    recommend_unrated_item_scorer import RecommendUnratedItemScorer
from azureml.designer.modules.recommenders.dnn.wide_and_deep.score.recommend_all_items_scorer import \
    RecommendAllItemScorer
from azureml.designer.modules.recommenders.dnn.common.entry_utils import params_loader
from azureml.designer.modules.recommenders.dnn.common.entry_param import Boolean


class RecommenderPredictionKind(Enum):
    RatingPrediction = "Rating Prediction"
    ItemRecommendation = "Item Recommendation"


class RecommendedItemSelection(Enum):
    FromAllItems = "From All Items"
    FromRatedItems = "From Rated Items (for model evaluation)"
    FromUnratedItems = "From Unrated Items (to suggest new items to users)"


class ScoreWideAndDeepRecommenderModule:
    @params_loader
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

    @staticmethod
    def set_inputs_name(test_interactions: InteractionDataset, training_interactions: InteractionDataset = None,
                        user_features: FeatureDataset = None, item_features: FeatureDataset = None):
        _INTERACTIONS_NAME = "Dataset to score"
        _USER_FEATURES_NAME = "User features"
        _ITEM_FEATURES_NAME = "Item features"
        _TRAINING_INTERACTIONS_NAME = "Training data"
        if test_interactions is not None:
            test_interactions.name = _INTERACTIONS_NAME
        else:
            ErrorMapping.verify_not_null_or_empty(x=test_interactions, name=_INTERACTIONS_NAME)
        if training_interactions is not None:
            training_interactions.name = _TRAINING_INTERACTIONS_NAME
        if user_features is not None:
            user_features.name = _USER_FEATURES_NAME
        if item_features is not None:
            item_features.name = _ITEM_FEATURES_NAME

    @params_loader
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
            return_ratings: Boolean = None,
            scored_data: str = None):
        module_logger.info(f"Update score params.")
        self.update_params(prediction_kind, recommended_item_selection, max_recommended_item_count,
                           min_recommendation_pool_size, return_ratings)
        self.set_inputs_name(test_interactions, training_interactions, user_features=user_features,
                             item_features=item_features)
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
        if scored_data is not None:
            save_data_frame_to_directory(save_to=scored_data, data=res, schema=DataFrameSchema.data_frame_to_dict(res))
        return res
