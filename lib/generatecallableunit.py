import torch
import torch.nn as nn
from lib.splitmodel import BasedSplitMethod
from lib.callableunit import CallableUnit


class BasedGenerateCallableUnit(object):
    """
    The base class of spliting model
    """
    def __init__(self, model, based_split_method: BasedSplitMethod):
        self.modules = model
        self._split_model_tool = based_split_method(model)

    def generate_callable_unit(self, method):
        """args: 
        method: split method
        """
        raise NotImplementedError()

    def split_model(self, *args, **kwargs):
        return self._split_model_tool.split_model(*args, **kwargs)


class GenerateCallableUnit(BasedGenerateCallableUnit):
    def __init__(self, model, based_split_method, is_recomputed, local_rank,
                 group, stage):
        super(GenerateCallableUnit, self).__init__(model, based_split_method)
        self.is_recomputed = is_recomputed
        self.local_rank = local_rank
        self.group = group
        self.stage = stage

    def generate_callable_unit(self, method):
        partitions = self.split_model(nonmean=False, method=method)
        num_partitions = len(partitions)
        num_stages = num_partitions // 2
        return CallableUnit(
            nn.ModuleList(
                [partitions[self.stage], partitions[self.stage + num_stages]]),
            num_stages, self.local_rank, self.is_recomputed, self.group)
