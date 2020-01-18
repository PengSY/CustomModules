from abc import abstractmethod
from enum import Enum
from azureml.designer.modules.recommenders.dnn.common.constants import TUPLE_SEP


class MiddleType:
    @abstractmethod
    def __new__(cls, *args, **kwargs):
        pass


class IntTuple(MiddleType):
    def __new__(cls, seq_str: str):
        integers = seq_str.split(TUPLE_SEP)
        try:
            integers = tuple([int(integer) for integer in integers])
        except TypeError:
            raise TypeError(f"Cannot convert {seq_str} to int tuple")
        return integers


class Boolean(MiddleType):
    def __new__(cls, bool_str: str):
        if bool_str not in ["True", "False"]:
            raise TypeError(f"Cannot convert {bool_str} to bool type")
        return bool_str == "True"


class DeepActivationSelection(Enum):
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


class RecommenderPredictionKind(Enum):
    RatingPrediction = "Rating Prediction"
    ItemRecommendation = "Item Recommendation"


class RecommendedItemSelection(Enum):
    FromAllItems = "From All Items"
    FromRatedItems = "From Rated Items (for model evaluation)"
    FromUnratedItems = "From Unrated Items (to suggest new items to users)"
