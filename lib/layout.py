from collections import defaultdict
from enum import Enum
import torch
import torch.distributed as dist
import torch.nn as nn
import numpy as np

GroupClass = Enum("GroupClass", ("G0", "G1"))

class BasedLayout(object):
    """
    based on rank and express network relationship between ranks, 
    input tensors name and output tensors name. provide data parallel group,
    model parallel group, adjacent group, receive_ranks, send_ranks, ranks_in_previous_stage,
    ranks_in_next_stage
    """
    def __init__(self, network_config, backend, local_rank, is_broadcast):
        self.backend = backend
        self.is_broadcast = is_broadcast
        self._local_rank = local_rank
        self._data_parallel_group = None
        self._previous_model_parallel_group = None
        self._next_model_parallel_group = None
        self._ranks_in_previous_stage = []
        self._ranks_in_next_stage = []
        self._ranks_in_stage = []
        self._stage = None
        self._num_stages = None
        self.network_config = network_config
        self.initialize(network_config)

    def initialize(self, network_config):
        raise NotImplementedError()

    def get_previous_model_parallel_group(self):
        raise NotImplementedError()

    def get_next_model_parallel_group(self):
        raise NotImplementedError()

    def get_data_parallel_group(self):
        raise NotImplementedError()

    @property
    def ranks_in_previous_stage(self):
        return self._ranks_in_previous_stage

    @property
    def ranks_in_next_stage(self):
        return self._ranks_in_next_stage

    @property
    def stage(self):
        return self._stage

    @property
    def ransk_in_stage(self):
        return self._ranks_in_stage

    @property
    def num_stages(self):
        return self._num_stages

    @property
    def local_rank(self):
        return self._local_rank

    def is_output_stage(self):
        return self._stage == self.num_stages - 1


class GlobalDataParallelLayout(BasedLayout):
    """
    only support global data parallel of model parallel
    args:
    network_config: data parallel and model parallel ranks configure
    backend: gloo or nccl. if is_broadcast = False, the backend must be gloo
    is_broadcast: bool, if is_broadcast = True, use dist.broadcast to communicate
    """
    def __init__(self, network_config, backend, local_rank,
                 is_broadcast=False):
        super(GlobalDataParallelLayout,
              self).__init__(network_config, backend, local_rank, is_broadcast)

    def initialize(self, network_config):
        # initialize group
        assert dist.is_available() and dist.is_initialized(), \
            "the distributed package is not available or process group is not initialized"
        data_parallel_groups = []
        for stage, info in network_config:
            data_parallel_groups.append(info["parallel"])
        model_parallel_groups = [[row[i] for row in data_parallel_groups]
                                 for i in range(len(data_parallel_groups[0]))]

        self._num_stages = len(data_parallel_groups)
        rank = dist.get_rank()
        # data parallel group
        for i, g in enumerate(data_parallel_groups):
            data_parallel_group = dist.new_group(g, backend="nccl")
            if rank in g:
                self._data_parallel_group = data_parallel_group
                self._stage = i
                self._ranks_in_stage = g

        # model parallel group
        for g in model_parallel_groups:
            L = len(g)
            next_group = None
            first_group = dist.new_group([g[(-1) % L], g[0 % L]],
                                         backend=self.backend)
            for i in range(L):
                previous_group = first_group if next_group is None else next_group
                next_group = dist.new_group(
                    [g[i % L], g[(i + 1) % L]],
                    backend=self.backend) if i + 1 < L else first_group
                if rank == g[i]:
                    self._previous_model_parallel_group = previous_group
                    self._next_model_parallel_group = next_group
                    self._ranks_in_previous_stage = [g[(i - 1) % L]]
                    self._ranks_in_next_stage = [g[(i + 1) % L]]

    def get_previous_model_parallel_group(self, tensor_name, group_class,
                                          direction):
        """it can be None"""
        return self._previous_model_parallel_group, self._previous_model_parallel_group

    def get_next_model_parallel_group(self, tensor_name, group_class,
                                      direction):
        """it can is None"""
        return self._next_model_parallel_group, self._next_model_parallel_group

    def get_data_parallel_group(self):
        assert self._data_parallel_group is not None, "warning! the data_parallel_group is None"
        return self._data_parallel_group


class DataModelParallelLayout(GlobalDataParallelLayout):
    """
    setup a group for the each output tensor
    """
    def __init__(self, network_config, backend, local_rank,
                 is_broadcast=False):
        super(DataModelParallelLayout, self).__init__(network_config, backend,
                                                      local_rank, is_broadcast)

    def initialize(self, network_config):
        assert dist.is_available() and dist.is_initialized(), \
            "the distributed package is not available or process group is not initialized"
        data_parallel_groups = []
        for stage, info in network_config:
            data_parallel_groups.append(info["parallel"])

        self._num_stages = len(data_parallel_groups)
        rank = dist.get_rank()
        # data parallel group
        for i, g in enumerate(data_parallel_groups):
            data_parallel_group = dist.new_group(g, backend="nccl")
            if rank in g:
                self._data_parallel_group = data_parallel_group
                self._stage = i
                self._ranks_in_stage = g

    def init_model_parallel_group(self, interface_param):
        """
        setup model parallel group for each tensor
        """
        self._previous_model_parallel_group = {}
        self._next_model_parallel_group = {}
        data_parallel_groups = []
        for stage, info in self.network_config:
            data_parallel_groups.append(info["parallel"])
        model_parallel_groups = [[row[i] for row in data_parallel_groups]
                                 for i in range(len(data_parallel_groups[0]))]

        rank = dist.get_rank()
        # model parallel group
        for g in model_parallel_groups:
            g = g + g

            L = len(g)
            next_group = [{}, {}]
            for i in range(L):
                previous_group = next_group if len(next_group[0]) > 0 else None
                next_group = [{}, {}]
                if i == L - 1:
                    next_group = None
                else:
                    for key in interface_param.get_output_keys(i):
                        next_group[0][key] = (dist.new_group(
                            [g[i], g[i + 1]], backend=self.backend),
                                              dist.new_group([g[i], g[i + 1]],
                                                             backend="gloo"))
                        next_group[1][key] = (dist.new_group(
                            [g[i], g[i + 1]], backend=self.backend),
                                              dist.new_group([g[i], g[i + 1]],
                                                             backend="gloo"))
                if rank == g[i]:
                    group_class = GroupClass.G0 if i < L // 2 else GroupClass.G1
                    self._previous_model_parallel_group[
                        group_class] = previous_group
                    self._next_model_parallel_group[group_class] = next_group
                    self._ranks_in_previous_stage = [g[(i - 1) % L]]
                    self._ranks_in_next_stage = [g[(i + 1) % L]]

    def build_connection_in_same_group(self):
        if not self.is_broadcast:
            return
        Direct = Enum("Direct", ("Next", "Previous"))

        def broadcast_sigal(groups, direction):
            # 0 : forward, 1: backward, previous + forward->forward recv
            # previous + backward -> backward send, next + forward -> forward send
            # next + backward -> backward recv
            for i, group_dict in enumerate(groups):
                for key in group_dict:
                    group, signal_group = group_dict[key]
                    rank = dist.get_rank()
                    if direction == Direct.Previous:
                        src_ranks = [
                            rank
                        ] if i != 0 else self.ranks_in_previous_stage
                    else:
                        src_ranks = [rank
                                     ] if i == 0 else self.ranks_in_next_stage
                    for src_rank in src_ranks:
                        dist.broadcast(torch.ones((1), ).cuda(self.local_rank),
                                       src_rank,
                                       group=group)
                        dist.broadcast(torch.ones((1), ),
                                       src_rank,
                                       group=signal_group)

        for group_class in (GroupClass.G0, GroupClass.G1):
            if self._previous_model_parallel_group is not None:
                groups = self._previous_model_parallel_group[group_class]
                if groups is not None:
                    broadcast_sigal(groups, Direct.Previous)
            if self._next_model_parallel_group is not None:
                groups = self._next_model_parallel_group[group_class]
                if groups is not None:
                    broadcast_sigal(groups, Direct.Next)

    def get_previous_model_parallel_group(self, tensor_name, group_class,
                                          direction):
        """it can be None"""
        assert isinstance(group_class,
                          GroupClass), "The group_class type error"
        assert direction in (0, 1), "the direction out of range"
        assert tensor_name in self._previous_model_parallel_group[group_class][
            direction], "The tensor name not exist!"
        return self._previous_model_parallel_group[group_class][direction][
            tensor_name]

    def get_next_model_parallel_group(self, tensor_name, group_class,
                                      direction):
        """it can be None"""
        assert isinstance(group_class,
                          GroupClass), "The group_class type error"
        assert direction in (0, 1), "the direction out of range"
        assert tensor_name in self._next_model_parallel_group[group_class][
            direction], "The tensor name not exist!"
        return self._next_model_parallel_group[group_class][direction][
            tensor_name]


class InterfaceParam(object):
    """
    input and output parameters in each stage, the input and output is the same for
    each stage.
    stage_keys = {input_name: (dim, dtype) or [(dim, dtype), ...], or None}
    """
    def __init__(self):
        self._g0_input_keys = []
        self._g1_input_keys = []
        self._g0_output_keys = []
        self._g1_output_keys = []
        self._tensor_tag = {}
        self._input_tensor_requires_grad_map = defaultdict(list)

    def get_output_keys(self, index):
        L = len(self._g0_output_keys)
        if index < L:
            return self._g0_output_keys[index]
        else:
            index %= L
            return self._g1_output_keys[index]

    @property
    def input_keys_g0(self):
        return self._g0_input_keys

    @property
    def input_keys_g1(self):
        return self._g1_input_keys

    @property
    def output_keys_g0(self):
        return self._g0_output_keys

    @property
    def output_keys_g1(self):
        return self._g1_output_keys

    def transform_data_format(self, input_dict):
        """
        transform training data  format into required data format
        """
        result = {}
        for key in input_dict:
            tensor = input_dict[key]
            if isinstance(tensor, torch.Tensor):
                result[key] = (len(tensor.shape), tensor.dtype)
            elif isinstance(tensor, (tuple, list)):
                temp = [None] * len(tensor)
                for i, t in enumerate(tensor):
                    if isinstance(t, torch.Tensor):
                        temp[i] = (len(t.shape), t.dtype)
                    else:
                        temp[i] = (None, None)
                result[key] = temp
            elif isinstance(tensor, bool):
                result[key] = (1, bool)
            else:
                result[key] = None

        return result

    def add_stage_input_keys(self, input_keys, which_group: GroupClass):
        """
        args:
        input_keys: {key: (dim, dtype) or [(dim, dtype), ...], or None}
        which_group: GroupClass[GroupClass.G0, GroupClass.G1]
        """
        assert isinstance(which_group,
                          GroupClass), "the type of which_group error"
        self.generate_requires_grad_map(which_group, **input_keys)
        if which_group == GroupClass.G0:
            self._g0_input_keys.append(self.transform_data_format(input_keys))
        else:
            self._g1_input_keys.append(self.transform_data_format(input_keys))

    def add_stage_output_keys(self, output_keys, which_group: GroupClass):
        """
        args:
        input_keys: {key: (dim, dtype) or [(dim, dtype), ...], or None}
        which_group: GroupClass[GroupClass.G0, GroupClass.G1]
        """
        assert isinstance(which_group,
                          GroupClass), "the type of which_group error"
        if which_group == GroupClass.G0:
            self._g0_output_keys.append(
                self.transform_data_format(output_keys))
        else:
            self._g1_output_keys.append(
                self.transform_data_format(output_keys))

    def get_tensor_tag(self, tensor_name, stage, model_group, is_forward,
                       is_send):
        """
        the received tensor is needed to be transformed into sent tensor
        """
        if len(self._tensor_tag) == 0:
            self.generate_tensor_tag()
        L = len(self._g1_input_keys)

        if is_forward and not is_send or not is_forward and is_send:
            if stage == 0:
                if model_group == GroupClass.G0:
                    return None
                else:
                    model_group = GroupClass.G0
            stage = (stage - 1) % L

        if stage == L - 1 and model_group == GroupClass.G1:
            return None

        tensor_full_name = ".".join(
            (tensor_name, str(stage), str(model_group)))
        assert tensor_full_name in self._tensor_tag, "the tensor {} does not exist!".format(
            tensor_full_name)
        return self._tensor_tag[tensor_full_name]

    def _generate_tensor_tag(self, tensor_keys, group, tensor_tag_start):
        L = len(tensor_keys)
        for stage in range(L):
            if stage != L - 1 or group != GroupClass.G1:
                for tensor_name in tensor_keys[stage]:
                    tensor_full_name = ".".join(
                        (tensor_name, str(stage), str(group)))
                    self._tensor_tag[tensor_full_name] = tensor_tag_start
                    tensor_tag_start += 1
        return tensor_tag_start

    def generate_tensor_tag(self):
        """
        the tensor tag is based on outputs of each stage, not inputs
        """
        tensor_tag_start = 1

        tensor_tag_start = self._generate_tensor_tag(self._g0_output_keys,
                                                     GroupClass.G0,
                                                     tensor_tag_start)
        tensor_tag_start = self._generate_tensor_tag(self._g1_output_keys,
                                                     GroupClass.G1,
                                                     tensor_tag_start)

    def generate_requires_grad_map(self, which_group, **input_tensors):
        """ the requires_grad_map is based on inputs of each stage, not outputs """
        cur_input = {}

        def get_tensor_type(inputs):
            if isinstance(inputs, torch.Tensor):
                return inputs.requires_grad
            elif isinstance(inputs, (tuple, list)):
                result = []
                for item in inputs:
                    result.append(get_tensor_type(item))
                return result
            else:
                return False

        for k, v in input_tensors.items():
            cur_input[k] = get_tensor_type(v)
        self._input_tensor_requires_grad_map[which_group].append(cur_input)

    def get_requires_grad(self,
                          which_group,
                          stage,
                          tensor_keys,
                          is_forward=True,
                          is_send=False):
        """ write from the receiving end """
        num_stages = len(self._input_tensor_requires_grad_map[which_group])
        assert stage < num_stages, f"the stage {stage} is larger than the total number of stages"
        if is_forward or is_send:
            return self._input_tensor_requires_grad_map[which_group][stage][
                tensor_keys]
        else:
            assert stage != num_stages - 1 or which_group != GroupClass.G1, "The last stage \
            of G1 is not required to receive backward gradient"

            stage += 1
            if stage == num_stages:
                which_group = GroupClass.G1
                stage %= num_stages
            assert tensor_keys in self._input_tensor_requires_grad_map[
                which_group][stage], f"the {tensor_keys} don't exist!"
            return self._input_tensor_requires_grad_map[which_group][stage][
                tensor_keys]
