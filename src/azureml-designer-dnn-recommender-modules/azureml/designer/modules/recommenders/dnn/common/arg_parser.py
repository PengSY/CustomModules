import yaml
import argparse


class SpecArgType:
    INT = "Int"
    MODE = "Mode"
    DATA_FRAME_DIRECTORY = "DataFrameDirectory"
    MODEL_DIRECTORY = "ModelDirectory"
    FLOAT = "Float"
    STRING = "String"
    BOOLEAN = "Boolean"


MIDDLE_TYPES = {
    SpecArgType.INT: int,
    SpecArgType.MODE: str,
    SpecArgType.DATA_FRAME_DIRECTORY: str,
    SpecArgType.MODEL_DIRECTORY: str,
    SpecArgType.FLOAT: float,
    SpecArgType.STRING: str,
    SpecArgType.BOOLEAN: str
}


class MiddleParam:
    def __init__(self,
                 internal_name=None,
                 name=None,
                 type=None,
                 value=None,
                 optional=False,
                 is_port=False,
                 parent_internal_name=None,
                 parent_value=None):
        self.internal_name = internal_name
        self.name = name
        self.type = type
        self.value = value
        self.optional = optional
        self.is_port = is_port
        self.parent_internal_name = parent_internal_name
        self.parent_value = parent_value


def get_params(data: list, internal_name_lookup: dict, parent_internal_name=None, parent_value=None):
    params = {}
    for input in data:
        param = MiddleParam()
        param.name = input['name']
        param.type = input['type']
        param.optional = input.get('optional', False)
        param.is_port = input.get('port', False)
        param.internal_name = internal_name_lookup[param.name]
        param.parent_internal_name = parent_internal_name
        param.parent_value = parent_value
        params[param.internal_name] = param
        if 'options' in input:
            for option in input['options']:
                if isinstance(option, dict):
                    parent_internal_name = internal_name_lookup[param.name]
                    parent_value = list(option.keys())[0]
                    params.update(get_params(list(option.values())[0], internal_name_lookup, parent_internal_name,
                                             parent_value))
    return params


def get_internal_name_lookup(spec: dict):
    data = spec['implementation']['container']['args']
    internal_names = data[0::2]
    names = data[1::2]
    lookup = {}
    for internal_name, name in zip(internal_names, names):
        name = list(name.values())[0]
        internal_name = internal_name[2:]
        lookup[name] = internal_name
    return lookup


def parse_spec(spec_file: str):
    with open(spec_file, "r") as f:
        spec = yaml.safe_load(f)
    internal_name_lookup = get_internal_name_lookup(spec)
    input_params = get_params(spec['inputs'], internal_name_lookup)
    output_params = get_params(spec['outputs'], internal_name_lookup)
    return input_params, output_params


def build_params(spec_file):
    input_port_params = dict()
    input_normal_params = dict()
    input_params, output_params = parse_spec(spec_file)
    for internal_name, param in input_params.items():
        if param.is_port:
            input_port_params[internal_name] = param
        else:
            input_normal_params[internal_name] = param
    return input_normal_params, input_port_params, output_params


def build_command_line(params):
    parser = argparse.ArgumentParser()
    for _, param in params.items():
        parser.add_argument('--' + param.internal_name, type=MIDDLE_TYPES[param.type], default=None)
    return parser


def parse_command_line_args(module_spec: str):
    input_normal_params, input_port_params, output_params = build_params(module_spec)
    return vars(build_command_line(input_normal_params).parse_known_args()[0]), vars(
        build_command_line(input_port_params).parse_known_args()[0]), vars(
        build_command_line(output_params).parse_known_args()[0])
