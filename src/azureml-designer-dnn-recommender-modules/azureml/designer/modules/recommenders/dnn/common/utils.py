import os
import pickle
from enum import Enum
from functools import wraps, partial
from azureml.studio.internal.error import ErrorMapping, InvalidLearnerError
from azureml.studio.core.io.data_frame_directory import load_data_frame_from_directory
from azureml.designer.modules.recommenders.dnn.common.arg_parser import build_params
from azureml.designer.modules.recommenders.dnn.common.constants import MODEL_NAME
from azureml.designer.modules.recommenders.dnn.common.dataset import Dataset
from azureml.designer.modules.recommenders.dnn.common.arg_parser import SpecArgType
from azureml.designer.modules.recommenders.dnn.common.wide_and_deep_model import WideAndDeepModel


def get_spec_dir():
    return os.path.join(os.path.abspath(__file__ + "/../../"), "module_specs")


def data_frame_directory_convert(arg, internal_type):
    if isinstance(arg.value, internal_type):
        arg.value.name = arg.name
        return arg.value
    if arg.type != SpecArgType.DATA_FRAME_DIRECTORY:
        raise TypeError(f"Cannot convert {arg.type} to Dataset type")
    df = load_data_frame_from_directory(load_from_dir=arg.value).data
    return internal_type(df=df, name=arg.name)


def mode_convert(arg, internal_type):
    if isinstance(arg.value, internal_type):
        return arg.value
    if arg.type != SpecArgType.MODE:
        raise TypeError(f"Cannot convert {arg.type} to {internal_type.__name__}")
    if isinstance(arg.value, internal_type):
        return arg.value
    return internal_type(arg.value)


def model_directory_convert(arg, internal_type):
    if isinstance(arg.value, internal_type):
        return arg.value
    if arg.type != SpecArgType.MODEL_DIRECTORY:
        raise TypeError(f"Cannot convert {arg.type} to WideAndDeepModel type")
    try:
        with open(os.path.join(arg.value, MODEL_NAME + '.pkl'), "rb") as f:
            model = pickle.load(f)
            if isinstance(model, WideAndDeepModel):
                model.model_dir = arg.value
    except ModuleNotFoundError:
        raise InvalidLearnerError(arg_name=arg.name)
    if not isinstance(model, internal_type):
        raise InvalidLearnerError(arg_name=arg.name)
    return model


def plain_convert(arg, internal_type):
    return internal_type(arg.value)


def get_convertor(internal_type):
    if issubclass(internal_type, Dataset):
        return partial(data_frame_directory_convert, internal_type=internal_type)
    elif issubclass(internal_type, WideAndDeepModel):
        return partial(model_directory_convert, internal_type=internal_type)
    elif issubclass(internal_type, Enum):
        return partial(mode_convert, internal_type=internal_type)
    else:
        return partial(plain_convert, internal_type=internal_type)


def check_required_params_not_null(params):
    for param in params.values():
        parent_internal_name = param.parent_internal_name
        if param.value is None:
            if param.optional:
                continue
            elif parent_internal_name is not None and params.get(parent_internal_name, None) != param.parent_value:
                continue
            else:
                ErrorMapping.verify_not_null_or_empty(x=param.value, name=param.name)


def convert_params(params, param_annotations):
    internal_params = {}
    for name, internal_type in param_annotations.items():
        if name == "return":
            continue
        if name == "mpi_support":
            communicator = params.get("Communicator", None)
            internal_params[name] = communicator is not None
        try:
            param = params[name]
        except KeyError:
            raise KeyError(f"Param {name} is not in module spec")
        if param.value is not None:
            internal_params[name] = get_convertor(internal_type)(param)
        else:
            internal_params[name] = None
    return internal_params


def set_param_values(params, name_value_map: dict):
    for internal_name, param in params.items():
        param.value = name_value_map.get(internal_name, None)
    return params


def update_required_params(normal_params, port_params, obj):
    required_params = port_params.copy()
    for internal_name, param in normal_params.items():
        if getattr(obj, internal_name, None) is None:
            required_params[internal_name] = param
    return required_params


def before_init(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        spec_file = os.path.join(get_spec_dir(), func.__module__.split('.')[-1] + '.yaml')
        normal_params, _, _ = build_params(spec_file)
        normal_params = set_param_values(normal_params, kwargs)
        check_required_params_not_null(normal_params)
        internal_params = convert_params(normal_params, func.__annotations__)
        return func(*args, **internal_params)

    return wrapper


def before_run(func):
    @wraps(func)
    def wrapper(obj, **kwargs):
        spec_file = os.path.join(get_spec_dir(), func.__module__.split('.')[-1] + '.yaml')
        normal_params, port_params, output_params = build_params(spec_file)
        normal_params = set_param_values(normal_params, kwargs)
        port_params = set_param_values(port_params, kwargs)
        output_params = set_param_values(output_params, kwargs)
        required_params = update_required_params(normal_params, port_params, obj)
        check_required_params_not_null(required_params)
        internal_params = convert_params({**port_params, **normal_params, **output_params}, func.__annotations__)
        return func(obj, **internal_params)

    return wrapper
