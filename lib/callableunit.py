from collections import deque, defaultdict, namedtuple
from enum import Enum
import logging as logger
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel.distributed import _find_tensors
from lib.utils import copy_params, set_params, get_rng_states, restore_rng_states

TORCH_VERSION = torch.__version__
logger = logger.getLogger(__name__)
CGroupIndex = namedtuple('CGroupIndex', ('G0', 'G1'))
GroupIndex = CGroupIndex(0, 1)
CParamVersion = namedtuple('CParamVersion', ('V0', 'V1'))
ParamVersion = CParamVersion(0, 1)


class InterfaceAdapter(object):
    """
    provide pre-processing and post-processing for forward and backward
    """
    def pre_processing_forward(self, **kwargs):
        raise NotImplementedError()

    def post_processing_froward(self, **kwargs):
        raise NotImplementedError()

    def pre_processing_backward(self, **kwargs):
        raise NotImplementedError()

    def post_processing_backward(self, **kwargs):
        raise NotImplementedError()


class DefaultInterfaceAdapter(InterfaceAdapter):
    def pre_processing_forward(self, **kwargs):
        return kwargs

    def post_processing_froward(self, **kwargs):
        return kwargs

    def pre_processing_backward(self, **kwargs):
        return kwargs

    def post_processing_backward(self, **kwargs):
        keys_set = kwargs.keys()
        keys_list = []
        for key in keys_set:
            temp = key.split(".")
            if len(temp) == 3:
                keys_list.append(temp)
        keys_list = sorted(keys_list)
        group = set()
        list_res = defaultdict(list)
        for v0, v1, v2 in keys_list:
            group.add(v0)
            list_res[v1].append(kwargs.pop(".".join((v0, v1, v2))))
        assert len(
            group) <= 1, "the number of group of backward gradient only is one"
        # remove the prefix G0 or G1
        for key in set(kwargs.keys()):
            _, old_tensor_name = key.split(".")
            kwargs[old_tensor_name] = kwargs.pop(key)
        kwargs.update(list_res)

        return kwargs

    def is_require_grad(self, *inputs):
        for i in inputs:
            if isinstance(i, torch.Tensor) and i.requires_grad:
                return True
            elif isinstance(i, (tuple, list)):
                return self.is_require_grad(*i)
            else:
                return False

    def unpack_require_grad_tensor(self, outputs, backward_grad):
        forward_output_key = set()
        for k, v in outputs.items():
            if self.is_require_grad(v):
                forward_output_key.add(k)

        def unpack_list(forward_output, backward_gradient, grad, *inputs):
            for i, item in enumerate(inputs):
                if isinstance(item, (tuple, list)):
                    unpack_list(forward_output, backward_gradient, grad[i],
                                *item)
                elif isinstance(item, torch.Tensor) and item.requires_grad:
                    forward_output.append(item)
                    backward_gradient.append(grad[i])

        forward_output, backward_gradient = [], []
        # the keys in order_key require gradients
        for key in forward_output_key:
            if isinstance(outputs[key], (tuple, list)):
                unpack_list(forward_output, backward_gradient,
                            backward_grad[key], *outputs[key])
            else:
                forward_output.append(outputs[key])
                backward_gradient.append(backward_grad[key])
        return forward_output, backward_gradient


class BasedCallableUnitInterface(nn.Module):
    """ the common base interface """
    def __init__(self, num_stages, adapter):
        super().__init__()
        self._adapter = adapter
        self.num_stages = num_stages

    def pre_processing_forward(self, **kwargs):
        """pre-processing for input data in the forward process"""
        return self._adapter.pre_processing_forward(**kwargs)

    def post_processing_froward(self, **kwargs):
        """post-processing for output data in the forward process"""
        return self._adapter.post_processing_froward(**kwargs)

    def pre_processing_backward(self, **kwargs):
        """adapter the data input format"""
        return self._adapter.pre_processing_backward(**kwargs)

    def post_processing_backward(self, **kwargs):
        """adapter the data output format"""
        return self._adapter.post_processing_backward(**kwargs)

    def _register_input_hook(self, pre_name, **kwargs):
        def record_input_gradient(input_name):
            def hook(grad):
                self.gradient_buffer[input_name] = grad

            return hook

        def register_hook_for_list(name, *inputs):
            for i, tensor in enumerate(inputs):
                input_name = "{}.{}".format(name, i)
                if isinstance(tensor, torch.Tensor) and tensor.requires_grad:
                    tensor.register_hook(record_input_gradient(input_name))

        for key, value in kwargs.items():
            in_name = pre_name + "." + key
            if isinstance(value, torch.Tensor) and value.requires_grad:
                value.register_hook(record_input_gradient(in_name))
            elif isinstance(value, (list, tuple)):
                register_hook_for_list(in_name, *value)

    def forward(self, **kwargs):
        kwargs = self.pre_processing_forward(**kwargs)
        if self.training:
            outputs = self._forward(**kwargs)
        else:
            outputs = self._forward_eval(**kwargs)
        outputs = self.post_processing_froward(**outputs)
        return outputs

    def backward(self, **kwargs):
        kwargs = self.pre_processing_backward(**kwargs)
        result = self._backward(**kwargs)
        outputs = self.post_processing_backward(**result)
        return outputs

    def _forward_eval(self, **kwargs):
        raise NotImplementedError()

    def _forward(self, **kwargs):
        raise NotImplementedError()

    def _backward(self, **kwargs):
        raise NotImplementedError()

    def _comm_backward(self, outputs, backward_grad):
        forward_output, backward_gradient = self._adapter.unpack_require_grad_tensor(
            outputs, backward_grad)
        self.gradient_buffer = {}
        torch.autograd.backward(forward_output, backward_gradient)

        return self.gradient_buffer

    def _last_backward(self, outputs):
        assert "losses" in outputs or "loss" in outputs, "the key losses is not included in the last outputs"
        self.gradient_buffer = {}
        gradient = torch.tensor(1.0 / self.num_stages).cuda()
        if "losses" in outputs:
            outputs["losses"].backward(gradient)
        else:
            outputs["loss"].backward(gradient)

        return self.gradient_buffer

    def get_gradient_data(self, **kwargs):
        data_provider = kwargs.get("data_provider")
        if data_provider is None:
            #logger.warn("The data_provider is None")
            return kwargs
        else:
            return next(data_provider)

    def _register_backward_reducer_hook(self, model, output):
        if TORCH_VERSION >= '1.7' and model.reducer._rebuild_buckets():
            logger.info("Reducer buckets have been rebuilt in this iteration.")
        if dist.get_world_size() > self.num_stages:
            # data parallel
            model.require_forward_param_sync = True
            if model.find_unused_parameters:
                model.reducer.prepare_for_backward(list(_find_tensors(output)))
            else:
                model.reducer.prepare_for_backward([])

    def data_parallel_gradient_reduce(self, model, output):
        if (self.backward_iter + 1) % self.num_stages == 0:
            self._register_backward_reducer_hook(model, output)


class CallableUnitInterface(BasedCallableUnitInterface):
    """
    define the callable interface
    """
    def __init__(self,
                 group_models,
                 num_stages,
                 local_rank=None,
                 is_recomputed=False,
                 groups=None,
                 adapter=DefaultInterfaceAdapter()):
        super(CallableUnitInterface, self).__init__(num_stages, adapter)
        self.is_recomputed = is_recomputed
        self.group_models = group_models
        if not isinstance(group_models, nn.ModuleList):
            self.group_models = nn.ModuleList(group_models)
        self._adapter = adapter
        # prepare for data parallel and intra-model parallel
        if groups is not None:
            assert dist.is_initialized()
            local_rank = dist.get_rank() % torch.cuda.device_count(
            ) if local_rank is None else local_rank
            for i, (g, model) in enumerate(zip(groups, self.group_models)):
                self.group_models[i] = nn.parallel.DistributedDataParallel(
                    self.group_models[i].cuda(local_rank),
                    device_ids=[local_rank],
                    output_device=local_rank,
                    process_group=g)
        # prepare for recompute
        self.buffer_g0 = [None, None]
        self.buffer_g0[ParamVersion.V0] = copy_params(
            self.group_models[GroupIndex.G0], clone=True)
        self.buffer_g0[ParamVersion.V1] = copy_params(
            self.group_models[GroupIndex.G0], clone=True)

        self.input_buffer_g0 = None
        self.input_buffer_g1 = None
        self.output_buffer_g0 = None
        self.output_buffer_g1 = None
        if is_recomputed:
            self.input_buffer_g0 = []
            self.input_buffer_g1 = []
        else:
            self.output_buffer_g0 = []
            self.output_buffer_g1 = []

        self.forward_iter = 0
        # delay the backward pass of group G0
        self.backward_iter = num_stages
        self.forward_version = 0
        self.backward_version = 0

        self.gradient_buffer = {}
        # used for backward in the G1

        self._lastest_version = 0

    def reset(self):
        self.forward_iter = 0
        self.backward_iter = self.num_stages
        self.forward_version = 0
        self.backward_version = 0
        self._lastest_version = 0

    def new_version_generated(self):
        self._lastest_version = 1 - self._lastest_version
        self.forward_version += 1

    @property
    def lastest_version(self):
        return self._lastest_version

    @property
    def get_forward_iter(self):
        return self.forward_iter

    @property
    def get_backward_iter(self):
        return self.backward_iter

    def get_old_version_id(self):
        return (self.forward_version - 1) % 2

    def get_cur_version_id(self):
        return self.forward_version % 2

    def _forward_g0(self, **kwargs):
        raise NotImplementedError()

    def _forward_g1(self, **kwargs):
        raise NotImplementedError()

    def _which_group(self, num_iter):
        result = (num_iter // self.num_stages) % 2 == 0
        return result

    def _forward(self, **kwargs):
        if self._which_group(self.forward_iter):
            result = self._forward_g0(**kwargs)
        else:
            result = self._forward_g1(**kwargs)
        self.forward_iter += 1
        return result

    def _forward_eval(self, **kwargs):
        group_id = kwargs.pop("group_id")
        self.forward_iter += 1
        with torch.no_grad():
            return self.group_models[group_id](**kwargs)

    def _backward_g0(self, **kwargs):
        raise NotImplementedError()

    def _backward_g1(self, **kwargs):
        raise NotImplementedError()

    def _backward(self, **kwargs):
        if self._which_group(self.backward_iter):
            result = self._backward_g0(**kwargs)
        else:
            result = self._backward_g1(**kwargs)
        self.backward_iter += 1
        return result


class CallableUnit(CallableUnitInterface):
    """
    encapsulate partition model. implement recomputing, data parallel, model parallel
    Args:
    groug_models: [G0, G1]
    num_stages: the running period of each group
    local_rank: local rank on the current device
    is_recomputed: to control whether the G1 to enable recompute
    group: the data parallel or intra-model parallel process group
    """
    def __init__(self,
                 group_models,
                 num_stages,
                 local_rank=None,
                 is_recomputed=False,
                 group=None,
                 adapter=DefaultInterfaceAdapter()):
        super(CallableUnit,
              self).__init__(group_models, num_stages, local_rank,
                             is_recomputed, group, adapter)

    def _forward_g0(self, **kwargs):
        if self.is_recomputed:
            rng_states = get_rng_states()
            self.input_buffer_g0.append((rng_states, kwargs))
            with torch.no_grad():
                set_params(self.group_models[GroupIndex.G0],
                           self.buffer_g0[self.get_cur_version_id()])
                return self.group_models[GroupIndex.G0](**kwargs)
        else:
            self._register_input_hook("G0", **kwargs)
            set_params(self.group_models[GroupIndex.G0],
                       self.buffer_g0[self.get_cur_version_id()])
            # cancel gradient reduce and add it at the end
            # of a gradient accumulation period by _register_backward_reducer_hook
            with self.group_models[GroupIndex.G0].no_sync():
                outputs = self.group_models[GroupIndex.G0](**kwargs)
            self.output_buffer_g0.append(outputs)
            return outputs

    def _forward_g1(self, **kwargs):
        if not self.is_recomputed:
            self._register_input_hook("G1", **kwargs)
            # cancel gradient reduce and add it at the end
            # of a gradient accumulation period _register_backward_reducer_hook
            with self.group_models[GroupIndex.G1].no_sync():
                result = self.group_models[GroupIndex.G1](**kwargs)
            self.output_buffer_g1.append(result)
            return result
        rng_states = get_rng_states()
        self.input_buffer_g1.append((rng_states, kwargs))
        with torch.no_grad():
            return self.group_models[GroupIndex.G1](**kwargs)

    def _backward_g0(self, **kwargs):
        if self.is_recomputed:
            rng_states, inputs = self.input_buffer_g0.pop(0)
            # load old version parameter
            set_params(self.group_models[GroupIndex.G0],
                       self.buffer_g0[self.get_old_version_id()])
            # recomputed
            self._register_input_hook("G0", **inputs)
            with restore_rng_states(rng_states), self.group_models[
                    GroupIndex.G0].no_sync():
                outputs = self.group_models[GroupIndex.G0](**inputs)
            outputs = self._adapter.post_processing_froward(**outputs)

        else:
            set_params(self.group_models[GroupIndex.G0],
                       self.buffer_g0[self.get_old_version_id()])
            outputs = self.output_buffer_g0.pop(0)

        self.data_parallel_gradient_reduce(self.group_models[GroupIndex.G0],
                                           outputs)
        kwargs = self.get_gradient_data(**kwargs)
        return self._comm_backward(outputs, kwargs)

    def _backward_g1(self, **kwargs):
        # do not need to set parameters in group1
        if self.is_recomputed:
            rng_states, inputs = self.input_buffer_g1.pop(0)
            self._register_input_hook("G1", **inputs)
            with restore_rng_states(rng_states), self.group_models[
                    GroupIndex.G1].no_sync():
                outputs = self.group_models[GroupIndex.G1](**inputs)
            outputs = self._adapter.post_processing_froward(**outputs)
        else:
            outputs = self.output_buffer_g1.pop(0)

        self.data_parallel_gradient_reduce(self.group_models[GroupIndex.G1],
                                           outputs)
        kwargs = self.get_gradient_data(**kwargs)
        order_key = kwargs.keys()
        if "losses" in order_key or "loss" in order_key:
            return self._last_backward(outputs)
        return self._comm_backward(outputs, kwargs)
