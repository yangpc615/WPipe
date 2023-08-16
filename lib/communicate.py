import os
from typing import Callable, SupportsInt
from queue import Queue
from collections import defaultdict, namedtuple
import threading
from enum import Enum
import logging as logger
import torch
import torch.nn as nn
import torch.distributed as dist
from lib.layout import InterfaceParam, BasedLayout, GroupClass

COMMETHOD = Enum("COMMETHOD", ("Broadcast", "Network", "NcclP2p"))
COMOP = Enum("COMOP", ("Send", "Recv"))
DIRECTION = Enum("DIRECTION", ("Forward", "Backward"))

logger = logger.getLogger(__name__)
COMMUNICATION_FP16 = False


class AutoFP16:
    def __init__(self,
                 loss_scale=None,
                 init_scale=2.**16,
                 scale_factor=2.,
                 scale_window=2000,
                 min_loss_scale=2.**(-16),
                 max_loss_scale=2.**24):
        if loss_scale == None:
            self.dynamic = True
            self._loss_scale = min(max_loss_scale, init_scale)
        else:
            self.dynamic = False
            self._loss_scale = loss_scale

        self._max_loss_scale = max_loss_scale
        self._min_loss_scale = min_loss_scale
        self._scale_seq_len = scale_window

    def _check_overflow(self, fp16_tensor):
        cpu_sum = float(fp16_tensor.float().sum())
        return (cpu_sum == float('inf') or cpu_sum == -float('inf')
                or cpu_sum != cpu_sum)

    def __call__(self, tensor):
        while True:
            _t = (tensor * self._loss_scale).half()
            overflow = self._check_overflow(_t)
            if not overflow:
                return _t, self._loss_scale
            if self._min_loss_scale == self._loss_scale:
                return tensor, 0
            if (self._min_loss_scale):
                self._loss_scale = max(self._min_loss_scale,
                                       self._loss_scale / 2.)
            else:
                self._loss_scale = self._loss_scale / 2.

    def unscale(self, tensor, scale_override=None):
        scale = scale_override or self._loss_scale
        if scale != 1.0:
            tensor.mul_(1.0 / scale)


def fp16_unscale(tensor_fp16, factor):
    AutoFP16(factor).unscale(tensor_fp16)


def tensor_to_fp16(tensor):
    return AutoFP16()(tensor)


def communicate_optimizer(tensor):
    if tensor.dtype in [torch.float32, torch.float64] and COMMUNICATION_FP16:
        return tensor_to_fp16(tensor)
    return tensor, 0


def initialize(master_addr, master_port, local_rank, rank, world_size,
               backend):
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    assert dist.get_world_size() == world_size
    logger.info(
        "Finished initializing process group; backend:{}, rank:{}, world_size:{}"
        .format(backend, rank, world_size))


def _recv_broadcast(src_rank, tensor_dims, tensor_dtype, sub_process_group,
                    signal_group):
    """receive tensor by dist.broadcast"""
    if tensor_dtype is None:
        return None
    assert dist.get_backend(
        signal_group) == "gloo", "tensor_shape communicated by cpu"
    tensor_shape = torch.zeros(tensor_dims, dtype=torch.int)
    dist.broadcast(tensor_shape, src_rank, group=signal_group)
    tensor_shape = tensor_shape.to(torch.int).tolist()

    if tensor_dtype == bool:
        received_tensor = torch.zeros(tensor_shape, dtype=torch.float32)
        dist.broadcast(received_tensor, src_rank, group=signal_group)
        return bool(received_tensor)

    received_tensor = torch.zeros(tensor_shape, dtype=tensor_dtype).cuda()
    dist.broadcast(received_tensor, src_rank, group=sub_process_group)
    return received_tensor


def _send_broadcast(tensor, src_rank, sub_process_group, signal_group):
    """ send tensor by dist.broadcast """
    tensor_shape = torch.Tensor(list(tensor.shape)).to(torch.int)
    assert dist.get_backend(
        signal_group) == "gloo", "tensor_shape communicated by cpu"

    dist.broadcast(tensor_shape, src_rank, group=signal_group)

    if tensor.numel() == 1 and isinstance(tensor.item(), bool):
        dist.broadcast(tensor.to(torch.float32), src_rank, group=signal_group)
        return
    tensor_clone = tensor.detach().clone().contiguous()
    dist.broadcast(tensor_clone, src_rank, group=sub_process_group)


def _recv_network(src_rank, tensor_dims, tensor_dtype, tensor_tag,
                  sub_process_group, *args):
    """ receive tensor by dist.recv """
    if tensor_dtype is None:
        return None

    # add flag dim
    tensor_shape = torch.zeros(tensor_dims + 1, dtype=torch.int)
    assert dist.get_backend(
        sub_process_group
    ) == "gloo", "the dist.send and dist.recv only support gloo backend"
    dist.recv(tensor_shape, src_rank, group=sub_process_group, tag=tensor_tag)

    tensor_shape = tensor_shape.to(torch.int).tolist()
    factor = tensor_shape[-1]
    tensor_shape = tensor_shape[:-1]

    _tensor_dtype = tensor_dtype if factor == 0 else torch.float16
    received_tensor = torch.zeros(
        tensor_shape,
        dtype=_tensor_dtype if tensor_dtype != bool else torch.float32)
    dist.recv(received_tensor,
              src_rank,
              group=sub_process_group,
              tag=tensor_tag)

    if tensor_dtype == bool:
        return tensor_dtype(received_tensor)

    received_tensor = received_tensor.cuda()
    if factor != 0:
        received_tensor = received_tensor.to(tensor_dtype)
        fp16_unscale(received_tensor, factor * 1.0)

    return received_tensor


def _send_network(tensor, dst_rank, tensor_tag, sub_process_group, *args):
    """ send tensor by dist.send """
    tensor, factor = communicate_optimizer(tensor)
    tensor_shape = torch.Tensor(list(tensor.shape) + [int(factor)]).to(
        torch.int)

    assert dist.get_backend(
        sub_process_group
    ) == "gloo", "the dist.send and dist.recv only support gloo backend"
    dist.send(tensor_shape, dst_rank, group=sub_process_group, tag=tensor_tag)

    if tensor.is_cuda:
        tensor = tensor.cpu()

    dist.send(tensor, dst_rank, group=sub_process_group, tag=tensor_tag)


def batch_send_recv(p2p_op_list):
    """ call dist.batch_isend_irecv synchronously """
    reqs = dist.batch_isend_irecv(p2p_op_list)
    for req in reqs:
        req.wait()


def _recv_ncclp2p(src_rank, tensor_dims, tensor_dtypes, tensor_tags,
                  sub_process_groups, signal_groups):
    """ implement recv by dist.batch_isend_irecv """

    shape_recv_list = []
    tensor_shapes = []

    def generate_op_list_shape(tensor_dim, tensor_tag, signal_group):
        if isinstance(tensor_dim, (int)):
            tensor_shape = torch.zeros(tensor_dim, dtype=torch.int)
            shape_recv_list.append(
                dist.P2POp(dist.irecv,
                           tensor_shape,
                           src_rank,
                           group=signal_group,
                           tag=tensor_tag))
            return tensor_shape
        elif isinstance(tensor_dim, (tuple, list)):
            return [generate_op_list_shape(t, tensor_tag) for t in tensor_dim]
        else:
            raise TypeError("tensor_dim type error in _recv_ncclp2p")

    for tensor_dim, tensor_tag, signal_group in zip(tensor_dims, tensor_tags,
                                                    signal_groups):
        tensor_shapes.append(
            generate_op_list_shape(tensor_dim, tensor_tag, signal_group))
    # receive the tensors shapes
    batch_send_recv(shape_recv_list)

    tensor_recv_list = []
    tensors = []

    def generate_op_list_tensor(tensor_shape, tensor_dtype, tag,
                                sub_process_group):
        if isinstance(tensor_shape, torch.Tensor):
            tensor_shape = tensor_shape.to(torch.int).tolist()
            received_tensor = torch.zeros(
                tensor_shape,
                dtype=tensor_dtype
                if tensor_dtype != bool else torch.float32).cuda()
            tensor_recv_list.append(
                dist.P2POp(dist.irecv,
                           received_tensor,
                           src_rank,
                           group=sub_process_group,
                           tag=tag))
            return received_tensor
        elif isinstance(tensor_shape, (tuple, list)):
            return tuple(
                generate_op_list_tensor(t, dtype, tag)
                for t, dtype in zip(tensor_shape, tensor_dtype))
        else:
            raise TypeError("tensor_shape type error in _recv_ncclp2p")

    for tensor_shape, tensor_dtype, tensor_tag, sub_process_group in zip(
            tensor_shapes, tensor_dtypes, tensor_tags, sub_process_groups):
        tensors.append(
            generate_op_list_tensor(tensor_shape, tensor_dtype, tensor_tag,
                                    sub_process_group))
    # receive the tensors
    batch_send_recv(tensor_recv_list)

    for i, tensor_dtype in enumerate(tensor_dtypes):
        if tensor_dtype == bool:
            tensors[i] = tensor_dtype(tensors[i].item())

    return tensors


def _send_ncclp2p(tensors, dst_rank, tensor_tags, sub_process_groups,
                  signal_groups):
    """ send tensors by dist.batch_isend_irecv """

    shape_send_list = []
    tensor_send_list = []

    def generate_op_list(tensor, tensor_tag, sub_process_group, signal_group):
        if isinstance(tensor, torch.Tensor):
            tensor_shape = torch.Tensor(list(tensor.shape)).to(torch.int)
            shape_send_list.append(
                dist.P2POp(dist.isend,
                           tensor_shape,
                           dst_rank,
                           group=signal_group,
                           tag=tensor_tag))
            tensor_send_list.append(
                dist.P2POp(dist.isend,
                           tensor.detach().clone().contiguous(),
                           dst_rank,
                           group=sub_process_group,
                           tag=tensor_tag))
        elif isinstance(tensor, (tuple, list)):
            for t in tensor:
                generate_op_list(t, tensor_tag)
        else:
            raise TypeError("Error! input mistake datatype into send_ncclp2p")

    for tensor, tag, sub_process_group, signal_group in zip(
            tensors, tensor_tags, sub_process_groups, signal_groups):
        generate_op_list(tensor, tag, sub_process_group, signal_group)
    # send tensors shapes
    batch_send_recv(shape_send_list)
    # send tensors
    batch_send_recv(tensor_send_list)


def _send(com_method: COMMETHOD, *args):
    assert isinstance(com_method, COMMETHOD), "the type of com_method error"
    if com_method == COMMETHOD.Broadcast:
        _send_broadcast(*args)
    elif com_method == COMMETHOD.Network:
        _send_network(*args)
    elif com_method == COMMETHOD.NcclP2p:
        _send_ncclp2p(*args)
    else:
        raise NotImplementedError("{} is not yet supported".format(com_method))


def _recv(com_method: COMMETHOD, *args):
    assert isinstance(com_method, COMMETHOD), "the type of com_method error"
    if com_method == COMMETHOD.Broadcast:
        return _recv_broadcast(*args)
    elif com_method == COMMETHOD.Network:
        return _recv_network(*args)
    elif com_method == COMMETHOD.NcclP2p:
        return _recv_ncclp2p(*args)
    else:
        raise NotImplementedError("{} is not yet supported".format(com_method))


def send_thread_func(queue, num_iterations, local_rank, com_method: COMMETHOD,
                     *send_args):
    if com_method == COMMETHOD.NcclP2p:
        return send_ncclp2p_thread_func(queue, num_iterations, local_rank,
                                        *send_args)
    torch.cuda.set_device(local_rank)
    for i in range(num_iterations):
        tensors = queue.get()
        if not isinstance(tensors, (list, tuple)):
            tensors = [tensors]
        for t in tensors:
            if t is None:
                continue
            elif isinstance(t, bool):
                t = torch.Tensor([t]).to(torch.bool)
            _send(com_method, t, *send_args)


def recv_thread_func(queue, num_iterations, local_rank, num_tensors,
                     com_method: COMMETHOD, *recv_args):
    if com_method == COMMETHOD.NcclP2p:
        return recv_ncclp2p_thread_func(queue, num_iterations, local_rank,
                                        num_tensors, *recv_args)
    torch.cuda.set_device(local_rank)
    for i in range(num_iterations):
        if num_tensors == 0:
            tensor = _recv(com_method, *recv_args)
            queue.put(tensor)
        else:
            tensors = [None] * num_tensors
            for c, dim, dtype in enumerate(zip(recv_args[1], recv_args[2])):
                args = (recv_args[0], dim, dtype) + recv_args[3:]
                tensors[c] = _recv(com_method, *args)
            queue.put(tensors)


def send_ncclp2p_thread_func(queues, num_iterations, local_rank, *send_args):
    torch.cuda.set_device(local_rank)
    for i in range(num_iterations):
        tensors = [queue.get() for queue in queues]
        send_tensors = []
        for t in tensors:
            if t is None:
                continue
            elif isinstance(t, bool):
                t = torch.Tensor([t]).to(torch.float32).cuda()
            send_tensors.append(t)
        _send(COMMETHOD.NcclP2p, send_tensors, *send_args)


def recv_ncclp2p_thread_func(queues, num_iterations, local_rank, num_tensors,
                             *recv_args):
    torch.cuda.set_device(local_rank)
    for i in range(num_iterations):
        tensors = _recv(COMMETHOD.NcclP2p, *recv_args)
        for queue, tensor in zip(queues, tensors):
            queue.put(tensor)


class CommunicationInterface(object):
    """
    define the necessary interface for activation transmission
    """
    def __init__(self, layout: BasedLayout, interface_info: InterfaceParam,
                 num_iterations: int, is_fp16: bool):
        assert dist.is_available() and dist.is_initialized(
        ), "the process group is not initialized"
        self._layout = layout
        self._interface_info = interface_info
        self.num_iterations = num_iterations
        global COMMUNICATION_FP16
        COMMUNICATION_FP16 = is_fp16

    @property
    def layout(self):
        return self._layout

    @property
    def interface_info(self):
        return self._interface_info

    def initialize(self, only_forward, eval_num_iters=None):
        com_method = COMMETHOD.Network
        if self._layout.is_broadcast:
            com_method = COMMETHOD.Broadcast
        elif self._layout.backend == "nccl":
            com_method = COMMETHOD.NcclP2p
        self.setup_queue(only_forward)
        logger.info("The queues in communication setup!")
        self.start_listen(com_method,
                          train=not only_forward,
                          eval_num_iters=eval_num_iters)
        logger.info("The communication threads have started!")

    def setup_queue(self, only_forward=False):
        """
        setup on queue for per tensor in forward / backward direction
        """
        self.forward_recv_queues = {
            GroupClass.G0: defaultdict(list),
            GroupClass.G1: defaultdict(list)
        }
        self.forward_send_queues = {
            GroupClass.G0: defaultdict(list),
            GroupClass.G1: defaultdict(list)
        }
        self.backward_recv_queues = {
            GroupClass.G0: defaultdict(list),
            GroupClass.G1: defaultdict(list)
        }
        self.backward_send_queues = {
            GroupClass.G0: defaultdict(list),
            GroupClass.G1: defaultdict(list)
        }
        stage = self._layout.stage

        input_keys_g0 = self._interface_info.input_keys_g0[stage]
        output_keys_g0 = self._interface_info.output_keys_g0[stage]
        input_keys_g1 = self._interface_info.input_keys_g1[stage]
        output_keys_g1 = self._interface_info.output_keys_g1[stage]

        def append_queue(in_out_keys,
                         dict_queues,
                         group,
                         depend_upstream,
                         is_forward,
                         stage,
                         only_forward=False):
            for key, value in in_out_keys.items():
                if value is not None and (is_forward or not only_forward):
                    # stage == 0, read data by DataLoader, but not for G1
                    if depend_upstream and (stage != 0
                                            or group == GroupClass.G1) \
                                       and (is_forward or
                                            self._interface_info.get_requires_grad(group, stage, key, False, True)):

                        for _ in range(
                                len(self._layout.ranks_in_previous_stage)):
                            dict_queues[group][key].append(Queue())
                    # the last stage, not necessary to send data in the forward
                    elif not depend_upstream and (
                            stage != self._layout.num_stages - 1
                            or group != GroupClass.G1) \
                            and (is_forward or
                                 self._interface_info.get_requires_grad(group, stage, key, False, False)):

                        for _ in range(len(self._layout.ranks_in_next_stage)):
                            dict_queues[group][key].append(Queue())

        append_queue(input_keys_g0, self.forward_recv_queues, GroupClass.G0,
                     True, True, stage, only_forward)
        append_queue(input_keys_g1, self.forward_recv_queues, GroupClass.G1,
                     True, True, stage, only_forward)
        append_queue(output_keys_g0, self.forward_send_queues, GroupClass.G0,
                     False, True, stage, only_forward)
        append_queue(output_keys_g1, self.forward_send_queues, GroupClass.G1,
                     False, True, stage, only_forward)
        append_queue(input_keys_g0, self.backward_send_queues, GroupClass.G0,
                     True, False, stage, only_forward)
        append_queue(input_keys_g1, self.backward_send_queues, GroupClass.G1,
                     True, False, stage, only_forward)
        append_queue(output_keys_g0, self.backward_recv_queues, GroupClass.G0,
                     False, False, stage, only_forward)
        append_queue(output_keys_g1, self.backward_recv_queues, GroupClass.G1,
                     False, False, stage, only_forward)

    def send(self, ):
        raise NotImplementedError()

    def recv(self, ):
        raise NotImplementedError()

    def start_thread(self, func, func_args):
        listener = threading.Thread(target=func, args=func_args)
        listener.start()

    def start_multi_threads(self, com_method, com_op, *args):
        if com_method != COMMETHOD.NcclP2p:
            for func, func_args in args:
                self.start_thread(func, func_args)
        elif len(args) > 0:
            func = args[0][0]
            func_args_list = [func_args for _, func_args in args]
            need_stack = [0, -1, -2, -3] if com_op == COMOP.Send else [
                0, 3, -1, -2, -3, -4, -5
            ]

            last_args = list(func_args_list[0])
            for i in need_stack:
                last_args[i] = tuple(item[i] for item in func_args_list)
            self.start_thread(func, tuple(last_args))
        else:
            logger.warn("The threads start args may be mistake!")

    def _get_call_args(self, index, tensor_name, model_group: GroupClass,
                       direction: DIRECTION, com_op: COMOP,
                       com_method: COMMETHOD):
        """ the boundary status judgement is needed to be implemented on the outside """
        CDirection = namedtuple("CDirection", ("Forward", "Backward"))
        Direction = CDirection(0, 1)
        if direction == DIRECTION.Forward and com_op == COMOP.Send:
            dst_rank = self._layout.ranks_in_next_stage[index]
            next_group, signal_group = self._layout.get_next_model_parallel_group(
                tensor_name, model_group, Direction.Forward)
            if com_method == COMMETHOD.Network or com_method == COMMETHOD.NcclP2p:
                tensor_tag = self._interface_info.get_tensor_tag(
                    tensor_name, self._layout.stage, model_group, True, True)
                assert tensor_tag is not None, "The tensor is not required to transfer because its tensor_tag is None"

                return dst_rank, tensor_tag, next_group, signal_group
            src_rank = dist.get_rank()
            return src_rank, next_group, signal_group

        if direction == DIRECTION.Forward and com_op == COMOP.Recv:
            src_rank = self._layout.ranks_in_previous_stage[index]
            input_keys = self._interface_info.input_keys_g0[self._layout.stage]
            if GroupClass.G1 == model_group:
                input_keys = self._interface_info.input_keys_g1[
                    self._layout.stage]

            dim_dtype = input_keys[tensor_name]
            if isinstance(dim_dtype, tuple):
                num_tensors = 0
                tensor_dims = dim_dtype[0]
                tensor_dtypes = dim_dtype[1]
            elif isinstance(dim_dtype, list):
                num_tensors = len(dim_dtype)
                tensor_dims = [dim for dim, _ in dim_dtype]
                tensor_dtypes = [dtype for _, dtype in dim_dtype]
            else:
                raise NotImplementedError(
                    "The {} dtype can not be sent or received".format(
                        dim_dtype))
            previous_group, signal_group = self._layout.get_previous_model_parallel_group(
                tensor_name, model_group, Direction.Forward)
            if com_method == COMMETHOD.Network or com_method == COMMETHOD.NcclP2p:
                tensor_tag = self._interface_info.get_tensor_tag(
                    tensor_name, self._layout.stage, model_group, True, False)
                assert tensor_tag is not None, "The tensor is not required to transfer because its tensor_tag is None"
                return num_tensors, (src_rank, tensor_dims, tensor_dtypes,
                                     tensor_tag, previous_group, signal_group)
            return num_tensors, (src_rank, tensor_dims, tensor_dtypes,
                                 previous_group, signal_group)

        if direction == DIRECTION.Backward and com_op == COMOP.Send:
            dst_rank = self._layout.ranks_in_previous_stage[index]
            previous_group, signal_group = self._layout.get_previous_model_parallel_group(
                tensor_name, model_group, Direction.Backward)
            if com_method == COMMETHOD.Network or com_method == COMMETHOD.NcclP2p:
                tensor_tag = self._interface_info.get_tensor_tag(
                    tensor_name, self._layout.stage, model_group, False, True)
                assert tensor_tag is not None, "The tensor is not required to transfer because its tensor_tag is None"
                return dst_rank, tensor_tag, previous_group, signal_group
            src_rank = dist.get_rank()
            return src_rank, previous_group, signal_group

        if direction == DIRECTION.Backward and com_op == COMOP.Recv:
            src_rank = self._layout.ranks_in_next_stage[index]
            input_keys = self._interface_info.output_keys_g0[
                self._layout.stage]
            if GroupClass.G1 == model_group:
                input_keys = self._interface_info.output_keys_g1[
                    self._layout.stage]

            dim_dtype = input_keys[tensor_name]
            if isinstance(dim_dtype, tuple):
                num_tensors = 0
                tensor_dims = dim_dtype[0]
                tensor_dtypes = dim_dtype[1]
            elif isinstance(dim_dtype, list):
                num_tensors = len(dim_dtype)
                tensor_dims = [dim for dim, _ in dim_dtype]
                tensor_dtypes = [dtype for _, dtype in dim_dtype]
            else:
                raise NotImplementedError(
                    "The {} dtype can not be sent or received".format(
                        dim_dtype))
            next_group, signal_group = self._layout.get_next_model_parallel_group(
                tensor_name, model_group, Direction.Backward)
            if COMMETHOD.Network == com_method or COMMETHOD.NcclP2p == com_method:
                tensor_tag = self._interface_info.get_tensor_tag(
                    tensor_name, self._layout.stage, model_group, False, False)
                assert tensor_tag is not None, "The tensor is not required to transfer because its tensor_tag is None"
                return num_tensors, (src_rank, tensor_dims, tensor_dtypes,
                                     tensor_tag, next_group, signal_group)
            return num_tensors, (src_rank, tensor_dims, tensor_dtypes,
                                 next_group, signal_group)

    def start_listen(self, com_method: COMMETHOD, train=True,
                     eval_num_iters=0):
        def get_num_iterations(direction: DIRECTION, com_op: COMOP,
                               model_group: GroupClass):
            num_iterations = self.num_iterations if train else eval_num_iters
            if self._layout.stage == 0:
                if direction == DIRECTION.Forward and com_op == COMOP.Recv and model_group == GroupClass.G0 or \
                   direction == DIRECTION.Backward and com_op == COMOP.Send and model_group == GroupClass.G0:
                    return 0
                return num_iterations
            if self._layout.stage == self._layout.num_stages - 1:
                if direction == DIRECTION.Forward and com_op == COMOP.Send and model_group == GroupClass.G1 or \
                   direction == DIRECTION.Backward and com_op == COMOP.Recv and model_group == GroupClass.G1:
                    return 0
                return num_iterations
            return num_iterations

        def start_helper_thread(name_queues, func, direction: DIRECTION,
                                com_op: COMOP, com_method: COMMETHOD):
            for model_group in name_queues:
                full_args_list = []
                for tensor_name in name_queues[model_group]:
                    for index in range(
                            len(name_queues[model_group][tensor_name])):
                        args = self._get_call_args(index, tensor_name,
                                                   model_group, direction,
                                                   com_op, com_method)
                        queue = name_queues[model_group][tensor_name][index]
                        if com_op == COMOP.Send:
                            num_iterations = get_num_iterations(
                                direction, com_op, model_group)
                            full_args = (queue, num_iterations,
                                         self._layout.local_rank,
                                         com_method) + args
                        else:
                            num_iterations = get_num_iterations(
                                direction, com_op, model_group)
                            num_tensors, recv_args = args
                            full_args = (queue, num_iterations,
                                         self._layout.local_rank, num_tensors,
                                         com_method) + recv_args
                        full_args_list.append((func, full_args))
                self.start_multi_threads(com_method, com_op, *full_args_list)

        start_helper_thread(self.forward_send_queues, send_thread_func,
                            DIRECTION.Forward, COMOP.Send, com_method)
        start_helper_thread(self.forward_recv_queues, recv_thread_func,
                            DIRECTION.Forward, COMOP.Recv, com_method)
        if train:
            start_helper_thread(self.backward_send_queues, send_thread_func,
                                DIRECTION.Backward, COMOP.Send, com_method)
            start_helper_thread(self.backward_recv_queues, recv_thread_func,
                                DIRECTION.Backward, COMOP.Recv, com_method)


class CommunicationHelper(CommunicationInterface):
    """
    send and receive the activation
    Args:
    layout: network relationshap between ranks
    interface_info: [{tensor_name:(dim, dtype) or [(dim, dtype), ...] or None}, ...]
    num_iterations: the number of iterations
    """
    def __init__(self,
                 layout: BasedLayout,
                 interface_info: InterfaceParam,
                 num_iterations: int,
                 is_fp16=True,
                 gather=None,
                 scatter=None):
        super(CommunicationHelper, self).__init__(layout, interface_info,
                                                  num_iterations, is_fp16)
        self._gather = gather
        self._scatter = scatter

    def gather(self, list_queue_tensor):
        result = []
        for q in list_queue_tensor:
            result.append(q.get())
        if self._gather is not None:
            return self._gather(result)
        if isinstance(result[0], torch.Tensor):
            return torch.cat(result)
        elif result[0] is None:
            return None
        elif isinstance(result[0], bool):
            return result[0]
        else:
            assert isinstance(result[0], (tuple, list)), "the type error!"
            return [
                torch.cat([row[i] for row in result])
                for i in range(len(result[0]))
            ]

    def scatter(self, tensors, split_num):
        if self._scatter is not None:
            return self._scatter(tensors)
        if isinstance(tensors, torch.Tensor):
            return tensors.chunk(split_num)
        elif tensors is None:
            return [None] * split_num
        elif isinstance(tensors, bool):
            return [tensors] * split_num
        else:
            assert isinstance(tensors, (tuple, list)), "the type error!"
            temp = [tensor.chunk(split_num) for tensor in tensors]
            return [[row[i] for row in temp] for i in range(split_num)]

    def recv(self, direction: DIRECTION, model_group: GroupClass):
        result = {}
        if direction == DIRECTION.Forward:
            for key in self.forward_recv_queues[model_group]:
                result[key] = self.gather(
                    self.forward_recv_queues[model_group][key])
                if self._interface_info.get_requires_grad(
                        model_group, self._layout.stage, key):
                    result[key] = result[key].requires_grad_()
        elif direction == DIRECTION.Backward:
            for key in self.backward_recv_queues[model_group]:
                if self._interface_info.get_requires_grad(
                        model_group, self._layout.stage, key, False):
                    result[key] = self.gather(
                        self.backward_recv_queues[model_group][key])
        else:
            raise ValueError("The {} choice not exist!".format(direction))
        return result

    def send(self, tensor_dict, direction: DIRECTION, model_group: GroupClass):
        """
        support data type: None, torch.Tensor, bool.
        default: the data type in tensor_dict is in the set of None, torch.Tensor, bool
        """
        if direction == DIRECTION.Forward:
            for key in tensor_dict:
                if tensor_dict[key] is None:
                    continue
                split_num = len(self.forward_send_queues[model_group][key])
                result = self.scatter(tensor_dict[key], split_num)
                for i in range(split_num):
                    self.forward_send_queues[model_group][key][i].put(
                        result[i])
        elif direction == DIRECTION.Backward:
            for key in tensor_dict:
                if tensor_dict[key] is None:
                    continue
                split_num = len(self.backward_send_queues[model_group][key])
                result = self.scatter(tensor_dict[key], split_num)
                for i in range(split_num):
                    self.backward_send_queues[model_group][key][i].put(
                        result[i])
        else:
            raise ValueError("The {} choice not exist!".format(direction))
