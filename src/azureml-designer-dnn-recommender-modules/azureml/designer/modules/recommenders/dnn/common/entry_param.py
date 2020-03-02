from azureml.designer.modules.recommenders.dnn.common.constants import TUPLE_SEP


class EntryParam(type):
    _LOADER = 'load'
    param_loaders = {}

    def __new__(mcs, name, bases, class_dict):
        cls = type.__new__(mcs, name, bases, class_dict)
        EntryParam.param_loaders[cls] = getattr(cls, EntryParam._LOADER)
        return cls

    @staticmethod
    def get_loader(param_type):
        if param_type in EntryParam.param_loaders:
            return EntryParam.param_loaders[param_type]
        return param_type


class IntTuple(metaclass=EntryParam):
    @staticmethod
    def load(seq_str: str):
        integers = seq_str.split(TUPLE_SEP)
        try:
            integers = tuple([int(integer) for integer in integers])
        except TypeError:
            raise TypeError(f"Cannot convert {seq_str} to int tuple")
        return integers


class Boolean(metaclass=EntryParam):
    @staticmethod
    def load(bool_str: str):
        if bool_str not in ["True", "False"]:
            raise TypeError(f"Cannot convert {bool_str} to bool type")
        return bool_str == "True"
