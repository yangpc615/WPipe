import torch
import inspect
import numpy as np
from transformers.modeling_bert import BertForSequenceClassification
from lib.splitmodel import IncludeCommonAttributeSplitMethod, SplitModel
from nlp.callbacks_eight import callback_eight
from nlp.callbacks_four import callback_four
from nlp.callbacks_n import callable_bert

callback_dict = {8: callback_eight, 4: callback_four}
NUM_PARTITIONS = 8


def get_callbacks(num_partitions):
    if num_partitions in callback_dict:
        return callback_dict[num_partitions]
    else:
        return callable_bert[:1] + callable_bert[1:2] * \
            (num_partitions - 2) + callable_bert[2:]


def wrapper(model, name, i):
    if name == "encoder":
        num_layers = len(getattr(model.bert, "encoder").layer)
        U = num_layers // NUM_PARTITIONS
        reminder = num_layers - U * NUM_PARTITIONS
        num_layers_list = [U] * NUM_PARTITIONS
        for index in range(reminder):
            num_layers_list[-index - 1] += 1
        num_layers_acc = [0] + np.cumsum(num_layers_list).tolist()
        return getattr(model.bert,
                       "encoder").layer[num_layers_acc[i]:num_layers_acc[i +
                                                                         1]]
    if name in ("dropout", "classifier", "num_labels"):
        return getattr(model, name)
    return getattr(model.bert, name)


def wrapper_nonmean(model, name, i):
    if name == "encoder":
        U = len(getattr(model.bert, "encoder").layer) // NUM_PARTITIONS
        part = NUM_PARTITIONS // 2
        partition_num = [U - 1] * part + [U + 1] * part
        return getattr(
            model.bert,
            "encoder").layer[sum(partition_num[:i]):sum(partition_num[:i + 1])]
    if name in ("dropout", "classifier", "num_labels"):
        return getattr(model, name)
    return getattr(model.bert, name)


class BertLargeUncastSplit(IncludeCommonAttributeSplitMethod):
    def __init__(self, model: BertForSequenceClassification):
        super().__init__(model.bert)
        self._out_model = model

    def split_model(self, *args, **kwargs):
        method = None
        if "method" in kwargs:
            method = kwargs.pop("method")
        if method == 8 or method is None:
            return self.split_eight(*args, **kwargs)
        elif method == 4:
            return self.split_four(*args, **kwargs)
        elif method % 2 == 0:
            return self.split_bert(method, wrapper=wrapper)
        else:
            raise ValueError(
                f"The method {method} not implemented in BertLargeUncastSplit")

    def split_four(self, *args, **kwargs):
        return self.split_bert(4, wrapper=wrapper)

    def split_eight(self, *args, **kwargs):
        nonmean = kwargs.get("nonmean")
        if not nonmean:
            return self.split_bert(8, wrapper=wrapper)
        else:
            return self.split_bert(8, wrapper=wrapper_nonmean)

    def split_bert(self, num_partitions, *args, **kwargs):
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
        parts0 = tuple(common_attributes) + ("embeddings", "encoder")
        parts.append(parts0)
        for _ in range(num_partitions - 2):
            parts.append(tuple(common_attributes) + ("encoder", ))
        parts_last = tuple(common_attributes) + (
            "encoder", "pooler", "dropout", "classifier", "num_labels")
        parts.append(parts_last)
        model_state_dict = ()
        name_map = {"encoder": "layer"}

        global NUM_PARTITIONS
        NUM_PARTITIONS = num_partitions
        wrapper = kwargs.get("wrapper")
        for part, callback in zip(parts, get_callbacks(num_partitions)):
            model_state_dict += ((part, callback), )
        # disable _gcn_enc data parallel
        results = SplitModel.split_model(self._out_model, model_state_dict,
                                         name_map, wrapper)
        return results
