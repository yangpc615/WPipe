import os
import time
import logging as logger
from typing import Optional, Tuple, List, Generator
from collections import OrderedDict, defaultdict
from contextlib import contextmanager
import gc
from tqdm import tqdm
import torch
from torch import Tensor
from torch.utils.data.dataloader import DataLoader
from torch.optim.lr_scheduler import LambdaLR
import torch.distributed as dist

logger = logger.getLogger(__name__)


def copy_params(module, clone=True):
    if isinstance(module, torch.nn.parallel.DistributedDataParallel):
        module = module.module

    state_dict = module.state_dict()
    for key in state_dict:
        state_dict[key] = state_dict[key].clone() if clone else state_dict[key]
    return state_dict


def set_params(module, state_dict):
    if isinstance(module, torch.nn.parallel.DistributedDataParallel):
        module = module.module

    cur_state_dict = module.state_dict()
    for key in state_dict:
        # Don't update running_mean and running_var; these should
        # accumulate normally.
        # mask might have a different shape, so don't copy it to
        # the module this way.
        if "running_" in key or "mask" in key:
            state_dict[key] = cur_state_dict[key]
    module.load_state_dict(state_dict)


def get_samper_args(network_config):
    """
    eg:
    network_config = [
        ("stage0", {"model": [0, 4], "parallel": [0]}),
        ("stage1", {"model": [1, 5], "parallel": [1]}),
        ("stage2", {"model": [2, 6], "parallel": [2]}),
        ("stage3", {"model": [3, 7], "parallel": [3]}),
    ]
    """
    stage, config = network_config[0]
    num_replicates = len(config["parallel"])
    rank = dist.get_rank()
    for stage, config in network_config:
        parallel = config["parallel"]
        if rank in parallel:
            index = parallel.index(rank)
            return num_replicates, index


class LossManager(object):
    """  """
    def __init__(self, interval):
        self._loss = 0
        self._num_iters = 0
        self._interval = interval
        self.losses_list = []
        self.lrs = defaultdict(list)

    def add(self, loss, *lrs):
        if isinstance(loss, torch.Tensor):
            loss = loss.item()
        self._loss += loss
        self._num_iters += 1
        if self._num_iters % self._interval == 0:
            time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            loss_mean = self._loss / self._num_iters
            self.losses_list.append(loss_mean)
            for i, lr in enumerate(lrs):
                self.lrs[f"lr{i}"].append(lr)
            logger.info(f"\n {time_str} --Loss-- {loss_mean} " +
                        ",".join("lr{}: {}".format(i, lr)
                                 for i, lr in enumerate(lrs)) + "\n")
            self._loss = 0
            self._num_iters = 0

    def save_data(self, name="losses.csv"):
        import pandas as pd
        if self.losses_list:
            data = pd.DataFrame(self.lrs)
            data["losses"] = self.losses_list
            name = name if isinstance(name, str) else "losses.csv"
            dirname = os.path.dirname(name)
            if not os.path.exists(dirname):
                os.makedirs(dirname)
            data.to_csv(name)
            logger.info(f"The data {self.losses_list} has been saved!")


def count_tensor_by_size():
    shape_nums = defaultdict(int)
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data')
                                        and torch.is_tensor(obj.data)):
                shape_nums[obj.size()] += 1
        except:
            pass
    for k, v in shape_nums.items():
        print(f"shape:{k}, times:{v}")


class MultiEpochDataLoader(object):
    """ support the multi-epoch data loader """
    def __init__(self, dataloader, epoch):
        self._dataloader = dataloader
        self.epoch = epoch

    def set_seed(self, seed):
        self._dataloader.sampler.set_epoch(seed)

    @property
    def batch_size(self):
        return self._dataloader.batch_size

    def __len__(self):
        return len(self._dataloader) * self.epoch

    def __iter__(self):
        for i in range(self.epoch):
            self._dataloader.sampler.set_epoch(i)
            for inputs in self._dataloader:
                yield inputs


class ProfileHelper(object):
    def __init__(self,
                 profile_dir_name='profile',
                 profile_step_num=4,
                 distributed=False,
                 stage=0):
        self.profile_step_num = profile_step_num
        if not os.path.isdir(profile_dir_name):
            os.makedirs(profile_dir_name)
        self.profile_dir_name = profile_dir_name
        if distributed:
            self.prefix = 'total_rank_' + str(dist.get_rank()) + '_'
        else:
            self.prefix = 'total_'
        self.prof = None
        self.delay_list = None
        self.stage = stage

    def set_delay(self, delay_list):
        self.delay_list = delay_list

    def start_check(self, step):
        delay = self.delay_list[self.stage]
        step -= delay
        if not step % self.profile_step_num and step >= 0:
            self.prof = torch.autograd.profiler.profile(
                use_cuda=True).__enter__()

    def end_check(self, step, epoch, clean=False):
        delay = self.delay_list[self.stage]
        step -= delay
        if self.prof and (
                clean or not (step + 1) % self.profile_step_num) and step >= 0:
            self.prof.__exit__(0, 0, 0)
            if clean:
                path = os.path.join(
                    self.profile_dir_name,
                    self.prefix + "epoch_%d_over.json" % (epoch))
            else:
                path = os.path.join(
                    self.profile_dir_name,
                    self.prefix + "epoch_%d_step_%d_to_%d.json" %
                    (epoch, step - self.profile_step_num + 1, step))
            self.prof.export_chrome_trace(path)


class DistributedProgressBar(object):
    """
    control the progress bar by rank
    """
    def __init__(self, total, rank, group=None):
        self._pbar = None
        cur_rank = 0
        if torch.distributed.is_available(
        ) and torch.distributed.is_initialized():
            cur_rank = dist.get_rank()
        if group is not None:
            cur_rank = dist.get_rank(group=group)
        if rank == cur_rank:
            self._pbar = tqdm(total=total)

    def update(self, step):
        if self._pbar:
            self._pbar.update(step)

    def close(self):
        if self._pbar:
            self._pbar.close()


def EvalDecorator(func):
    def wrapper_func(*args, **kwargs):
        with torch.no_grad():
            return func(*args, **kwargs)

    return wrapper_func


def distributed_concat(tensor: "torch.Tensor",
                       group,
                       local_rank,
                       num_total_examples: Optional[int] = None
                       ) -> torch.Tensor:
    try:
        if isinstance(tensor, (tuple, list)):
            return type(tensor)(
                distributed_concat(t, group, num_total_examples)
                for t in tensor)
        if dist.get_backend(group=group) == "nccl" and not tensor.is_cuda:
            tensor = tensor.cuda(local_rank)
        output_tensors = [
            tensor.clone()
            for _ in range(torch.distributed.get_world_size(group=group))
        ]
        torch.distributed.all_gather(output_tensors, tensor, group=group)
        concat = torch.cat(output_tensors, dim=0)

        # truncate the dummy elements added by SequentialDistributedSampler
        if num_total_examples is not None:
            concat = concat[:num_total_examples]
        return concat
    except AssertionError:
        raise AssertionError("Not currently using distributed training")


def nested_detach_cpu(tensors):
    if isinstance(tensors, (tuple, list)):
        return type(tensors)(t.detach().cpu() for t in tensors)
    assert isinstance(tensors, torch.Tensor), "type error! expect torch.Tensor"
    return tensors.detach().cpu()


def get_rng_states() -> None:
    """:meth:`forward` captures the current PyTorch's random number
    generator states at CPU and GPU to reuse in backward`.
    """
    cpu_rng_state = torch.get_rng_state()

    if torch.cuda.is_available():
        gpu_rng_state = torch.cuda.get_rng_state(torch.cuda.current_device())
    else:
        gpu_rng_state = None

    return cpu_rng_state, gpu_rng_state


@contextmanager
def restore_rng_states(rng_states: Tuple["cpu_rng_state", "gpu_rng_state"],
                       ) -> Generator[None, None, None]:
    """:meth:`Recompute in backward` restores the random number generator states
    captured by :func:`get_rng_states` within its context.
    """
    cpu_rng_state, gpu_rng_state = rng_states

    gpu_devices: List[torch.device] = []
    current_device = torch.cuda.current_device()
    if torch.cuda.is_available():
        gpu_devices.append(current_device)

    with torch.random.fork_rng(gpu_devices):
        torch.set_rng_state(cpu_rng_state)
        if gpu_rng_state is not None:
            torch.cuda.set_rng_state(gpu_rng_state, current_device)
        yield


class BatchNorm2dClone(torch.nn.BatchNorm2d):
    """
    clone running_mean and running_var
    """
    def forward(self, input: Tensor) -> Tensor:
        if self.running_mean is not None:
            self.running_mean = self.running_mean.clone()
        if self.running_var is not None:
            self.running_var = self.running_var.clone()
        return super(BatchNorm2dClone, self).forward(input)


def nested_concat(tensors, new_tensors, dim=0):
    "Concat the `new_tensors` to `tensors` on `dim`. Works for tensors or nested list/tuples of tensors."
    assert type(tensors) == type(
        new_tensors
    ), f"Expected `tensors` and `new_tensors` to have the same type but found {type(tensors)} and {type(new_tensors)}."
    if isinstance(tensors, (list, tuple)):
        return type(tensors)(nested_concat(t, n, dim)
                             for t, n in zip(tensors, new_tensors))
    return torch.cat((tensors, new_tensors), dim=dim)


def generate_network_config():
    pass


class ThroughPutCalculate(object):
    """  """
    def __init__(self, interval, data_parallel_size, batch_size):
        self._count = 0
        self._start_time = None
        self._throughput = 0
        self._interval = interval
        self._data_parallel_size = data_parallel_size
        self._batch_size = batch_size

    def update(self, count=1):
        self._count += 1
        if self._start_time is None:
            self._start_time = time.time()
            self._count = 0
        if self._count >= self._interval:
            end_time = time.time() - self._start_time
            self._throughput = (self._data_parallel_size * self._batch_size *
                                self._count) / end_time

    def save_data(self, name="throughput.txt"):
        if self._throughput != 0:
            with open(name, 'a+') as f:
                f.writelines(f"throughput:{self._throughput}\n")


def clip_input(inputs, batch_size=2):
    for key in inputs:
        if isinstance(inputs[key], torch.Tensor):
            inputs[key] = inputs[key][:batch_size]


def get_fixed_schedule_without_warmup(optimizer, *args, **kwargs):
    def lr_lambda(current_step: int):
        return 1.0

    return LambdaLR(optimizer, lr_lambda, -1)


class CalculateMemory(object):
    def __init__(self, count=2, rank=None):
        self._rank = rank
        if rank is None and dist.is_available() and dist.is_initialized():
            self._rank = dist.get_rank()
        else:
            self._rank = 0
        self._max_mean_memory = 0
        self._count = 0
        self._max_count = count

    def calculate_memory(self):
        if self._count > self._max_count:
            return
        self._count += 1
        num_gpus = torch.cuda.device_count()
        memory_list = []
        for i in range(num_gpus):
            m = torch.cuda.max_memory_allocated(i) / (1024 * 1024.)
            memory_list.append(m)
        total = sum(memory_list)
        if dist.is_available() and dist.is_initialized():
            total_tensor = torch.Tensor([total]).cuda()
            dist.reduce(total_tensor, 0)
            total = total_tensor.item()
        if self._rank == 0:
            single_gpu = total / num_gpus
            self._max_mean_memory = max(self._max_mean_memory, single_gpu)

    def save_data(self, name="memory.txt"):
        if self._max_mean_memory != 0:
            save_dir = os.path.dirname(name)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            with open(name, 'a+') as fp:
                fp.writelines(f"memory:{self._max_mean_memory}\n")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())
