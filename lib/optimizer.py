from collections import OrderedDict
import torch
from torch.optim.optimizer import Optimizer
import torch.optim as optim
from lib.callableunit import CallableUnit, GroupIndex


class DoubleParametersOptimizer(Optimizer):
    """ Wn+1 = Wn - lr * d(Wn-1) """
    def __init__(self, optimizer: torch.optim.Optimizer,
                 lr_scheduler: torch.optim.lr_scheduler.LambdaLR,
                 module: torch.nn.Module):
        self._base_optimizer = optimizer
        self._lr_scheduler = lr_scheduler
        self._module = module
        self._update_keys = set()

    def load_new_parameters(self, new_parameters: OrderedDict):
        cur_state_dict = self._module.state_dict()
        order_dict = OrderedDict(self._module.named_parameters())
        for key in cur_state_dict:
            if "running_" in key or "mask" in key or key not in order_dict or not order_dict[
                    key].requires_grad:
                new_parameters[key] = cur_state_dict[key]
            else:
                self._update_keys.add(key)
        self._module.load_state_dict(new_parameters)

    def step(self, closure=None):
        loss = self._base_optimizer.step(closure=closure)
        self._lr_scheduler.step()
        return loss

    def set_new_parameters(self, old_parameters: OrderedDict):
        cur_state_dict = self._module.state_dict()
        for key in self._update_keys:
            old_parameters[key] = cur_state_dict[key].clone()

    def zero_grad(self):
        self._base_optimizer.zero_grad()


class DoubleGroupsOptimizer(Optimizer):
    def __init__(self,
                 optimizer_g0: torch.optim.Optimizer,
                 optimizer_g1: torch.optim.Optimizer,
                 lr_scheduler_g0: torch.optim.lr_scheduler.LambdaLR,
                 lr_scheduler_g1: torch.optim.lr_scheduler.LambdaLR,
                 callableunit: CallableUnit,
                 gradient_acc_interval=1):
        self._callable_unit = callableunit
        module = callableunit.group_models[GroupIndex.G0]
        if isinstance(module, torch.nn.parallel.DistributedDataParallel):
            module = module.module
        self._optimizer_g0 = DoubleParametersOptimizer(optimizer_g0,
                                                       lr_scheduler_g0, module)
        self._optimizer_g1 = optimizer_g1
        self._lr_scheduler_g0 = lr_scheduler_g0
        self._lr_scheduler_g1 = lr_scheduler_g1
        self.gradient_acc_interval = gradient_acc_interval
        self.zero_grad_count = 0
        self.buffer_g0 = self._callable_unit.buffer_g0

    def zero_grad(self):
        if self.zero_grad_count % self.gradient_acc_interval == 0:
            if self._callable_unit._which_group(
                    self._callable_unit.backward_iter):
                self._optimizer_g0.zero_grad()
            else:
                self._optimizer_g1.zero_grad()
        self.zero_grad_count += 1

    def get_lr(self):
        lr_g0 = self._lr_scheduler_g0.get_last_lr()[0]
        lr_g1 = self._lr_scheduler_g1.get_last_lr()[0]
        return lr_g0, lr_g1

    def step(self, closure=None):
        if self._callable_unit.backward_iter % self.gradient_acc_interval == 0:
            if self._callable_unit._which_group(
                    self._callable_unit.backward_iter - 1):
                self._optimizer_g0.load_new_parameters(
                    self.buffer_g0[self._callable_unit.lastest_version])
                self._optimizer_g0.step(closure=closure)
                self._optimizer_g0.set_new_parameters(
                    self.buffer_g0[1 - self._callable_unit.lastest_version])
                self._callable_unit.new_version_generated()
                self._optimizer_g0.zero_grad()
            else:
                self._optimizer_g1.step(closure=closure)
                self._lr_scheduler_g1.step()
                self._optimizer_g1.zero_grad()
