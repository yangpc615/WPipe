import torch
import numpy as np
import inspect
from lib.splitmodel import IncludeCommonAttributeSplitMethod, SplitModel
from cv.callbacks import resnext_callbacks

PART_NUM = None
NUM_PARTITIONS = None


def wrapper(model, name, i):
    def get_layers():
        acc_layers = np.cumsum(PART_NUM)
        # -1: delete the conv1 layer,
        num_pre_layer = 1
        num_other_layer = 1
        start = acc_layers[i - 1] - num_pre_layer if i > 0 else 0
        end = acc_layers[
            i] - num_pre_layer if i != NUM_PARTITIONS - 1 else acc_layers[
                i] - num_other_layer
        if start == end:
            return None
        results = []
        for j in range(4):
            cur_layer = getattr(model, f"layer{j+1}")
            layer_len = len(cur_layer)
            start -= layer_len
            end -= layer_len
            if start < 0 and end > 0:
                results.append(cur_layer[start:])
                start = 0
            elif end <= 0:
                end += layer_len
                results.append(cur_layer[start:end])
                break

        return torch.nn.Sequential(*results)

    if name == "layer":
        return get_layers()
    if name == "loss_func":
        return torch.nn.CrossEntropyLoss()
    return getattr(model, name)


class ResnextSplit(IncludeCommonAttributeSplitMethod):
    def __init__(self, model):
        super().__init__(model)

    def split_model(self, *args, **kwargs):
        method = None
        if "method" in kwargs:
            method = kwargs.pop("method")
        if method == 8 or method is None:
            return self.split_eight(*args, **kwargs)
        elif method == 4:
            return self.split_four(*args, **kwargs)
        elif method == 2:
            return self.split_two(*args, **kwargs)
        elif method % 2 == 0:
            return self.split_resnext(method, wrapper=wrapper)
        else:
            raise ValueError(f"The method {method} not implemented in ResnextSplit")

    def split_four(self, *args, **kwargs):
        return self.split_resnext(4, ratio=None, wrapper=wrapper)

    def split_eight(self, *args, **kwargs):
        return self.split_resnext(8, wrapper=wrapper)

    def split_two(self, *args, **kwargs):
        return self.split_resnext(2, ratio=None, wrapper=wrapper)

    def split_resnext(self, num_partitions, ratio=None, *args, **kwargs):
        based_methods = inspect.getmembers(torch.nn.Module(),
                                           predicate=inspect.ismethod)
        based_methods = set(i for i, _ in based_methods)
        all_methods = inspect.getmembers(self.module,
                                         predicate=inspect.ismethod)
        all_methods = set(i for i, _ in all_methods)
        inner_method = all_methods - based_methods
        self.set_inner_methods(list(inner_method))
        common_attributes = set(self._get_common_attribute())

        parts = []
        parts0 = tuple(common_attributes) + ("conv1", "bn1", "relu", "maxpool",
                                             "layer")
        parts.append(parts0)
        for _ in range(num_partitions - 2):
            parts.append(tuple(common_attributes) + ("layer", ))
        parts_last = tuple(common_attributes) + ("layer", "avgpool", "fc",
                                                 "loss_func")
        parts.append(parts_last)

        layer_num = [
            len(self.module.layer1),
            len(self.module.layer2),
            len(self.module.layer3),
            len(self.module.layer4)
        ]
        layer_len = sum(layer_num)
        layer_len += 1
        if ratio is None:
            U = layer_len // num_partitions
            remainder = layer_len - U * num_partitions
            part_num = [U] * num_partitions
        else:
            part_num = [0] * num_partitions
            for i, r in enumerate(ratio):
                part_num[i] = int(layer_len * r)
            remainder = layer_len - sum(part_num)
        for i in range(remainder):
            part_num[-i - 1] += 1
        global PART_NUM
        PART_NUM = part_num
        global NUM_PARTITIONS
        NUM_PARTITIONS = num_partitions
        model_state_dict = ()

        wrapper = kwargs.get("wrapper")
        for part in parts:
            model_state_dict += ((part, resnext_callbacks), )
        # disable _gcn_enc data parallel
        results = SplitModel.split_model(self.module,
                                         model_state_dict,
                                         wrapper=wrapper)
        return results
