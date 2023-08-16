from typing import List, Dict, Any, Callable
from collections import OrderedDict
import torch
from torch import nn


class BasedModelUnit(nn.Module):
    """
    divided model
    args:
    callback: is executor for forward process,
        its format: func(self, args), if train and test are different, you need to
        solve it in callback.
    kwargs:
        running modules and parameters.
    """
    def __init__(self, callback, **kwargs):
        super().__init__()
        for k, v in kwargs.items():
            setattr(self, k, v)
        self._callback = callback

    def _forward(self, *inputs, **kwargs):
        return self._callback(self, *inputs, **kwargs)

    def forward_train(self, *inputs, **kwargs):
        return self._forward(*inputs, **kwargs)

    def forward_test(self, *inputs, **kwargs):
        return self._forward(*inputs, **kwargs)

    def forward(self, *inputs, **kwargs):
        if self.training:
            return self.forward_train(*inputs, **kwargs)
        else:
            return self.forward_test(*inputs, **kwargs)


class SplitModel:
    """
    the common model division method
    args:
    model: the main model
    model_state_dict: ((module_names, callback),..., (module_names, callback))
    """
    @staticmethod
    def split_model(model: nn.Module, model_state_dict: tuple, name_map: dict = None, wrapper: Callable = None):
        partitions = []
        for i, (module_name, callback) in enumerate(model_state_dict):
            models_dict = OrderedDict()
            for name in module_name:
                new_name = name
                if name_map is not None and name in name_map:
                    new_name = name_map[name]
                models_dict[new_name] = getattr(model, name) if wrapper is None else wrapper(model, name, i)
            partitions.append(BasedModelUnit(callback, **models_dict))
        return partitions


class BasedSplitMethod(object):
    """
    based split method
    """
    def __init__(self, model):
        self.module = model

    def split_model(self, *args, **kwargs):
        raise NotImplementedError()


class IncludeCommonAttributeSplitMethod(BasedSplitMethod):
    def __init__(self, model):
        super().__init__(model)
        self._exclude_attribute = ["_parameters", "_buffers", "_modules", "_backward_hooks",
                                   "_forward_hooks", "_forward_pre_hooks", "_state_dict_hooks",
                                   "_load_state_dict_pre_hooks", '__class__', '__delattr__', '__dict__',
                                   '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__',
                                   '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__lt__',
                                   '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__',
                                   '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__']
        self._inner_methods = []

    def set_inner_methods(self, funcs):
        self._inner_methods = funcs

    def _is_common_attribute(self, value: dict):
        result = True
        if isinstance(value, (tuple, list)):
            for item in value:
                result &= self._is_common_attribute(item)
            return result
        elif isinstance(value, dict):
            for item in value.values():
                result &= self._is_common_attribute(item)
            return result
        elif isinstance(value, (torch.nn.Parameter, torch.nn.Module)):
            return False
        else:
            return result

    def _get_common_attribute(self):
        common_attribute = []
        for k, v in self.module.__dict__.items():
            if k not in self._exclude_attribute and k not in self.module._modules:
                if self._is_common_attribute(v):
                    common_attribute.append(k)
        common_attribute += self._inner_methods
        return common_attribute
