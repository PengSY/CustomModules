import argparse
from functools import wraps
from azureml.designer.modules.recommenders.dnn.common.entry_param import EntryParam


def build_cli_args(func):
    parser = argparse.ArgumentParser()
    for p_name, p_type in func.__annotations__.items():
        if p_name != "return":
            parser.add_argument(f'--{p_name}', default=None)

    kwargs = vars(parser.parse_known_args()[0])
    return kwargs


def params_loader(func):
    @wraps(func)
    def wrapper(obj, **kwargs):
        for p_name, p_type in func.__annotations__.items():
            if isinstance(kwargs.get(p_name, None), p_type):
                continue
            if kwargs.get(p_name, None) is not None:
                kwargs[p_name] = EntryParam.get_loader(p_type)(kwargs[p_name])
        return func(obj, **kwargs)

    return wrapper
