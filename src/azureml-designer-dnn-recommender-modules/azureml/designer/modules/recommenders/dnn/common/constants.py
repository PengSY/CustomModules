from azureml.studio.core.data_frame_schema import ColumnTypeName

INTERACTIONS_USER_COL = 0
INTERACTIONS_ITEM_COL = 1
INTERACTIONS_RATING_COL = 2

USER_INTERNAL_KEY = "User"
ITEM_INTERNAL_KEY = "Item"
RATING_INTERNAL_KEY = "Rating"
FEATURE_INTERNAL_SUFFIX = "_feat"

FEATURES_ID_COL = 0

USER_ITEM_INTERACTIONS_DATASET = "User item interactions dataset"
ITEM_FEATURE_DATASET = "Item feature dataset"
USER_FEATURE_DATASET = "User feature dataset"

VALID_FEATURE_TYPE = (ColumnTypeName.NUMERIC, ColumnTypeName.STRING, ColumnTypeName.CATEGORICAL, ColumnTypeName.BINARY)

RANDOM_SEED = 42

MODEL_NAME = "model"
PARQUET_DATASET = "data.dataset.parquet"

TUPLE_SEP = ","
