import torch
from torch.utils.data.dataloader import DataLoader
from lib.communicate import CommunicationHelper, DIRECTION, GroupClass
from lib.utils import nested_detach_cpu


class BasedUnifiedDataProvider(object):
    """
    Unify communication data and DataLoader data
    """
    def __init__(self,
                 communication_helper: CommunicationHelper,
                 data_loader: DataLoader,
                 eval_data_loader: DataLoader = None,
                 test_data_loader: DataLoader = None):
        self._communication_helper = communication_helper
        # train, eval and test data loader
        self._eval_data_loader = eval_data_loader
        self._test_data_loader = test_data_loader
        self._train_data_loader = data_loader
        self._dataset_length = len(data_loader)
        self._data_loader = iter(self._train_data_loader)

        self.stage = self._communication_helper.layout.stage
        self.num_stages = self._communication_helper.layout.num_stages
        self.forward_iter = 0
        # delay the backward of group0
        self.backward_iter = self.num_stages
        self.num_warmup_macrobatches = self.num_stages - 1 - self.stage
        self.main_iter = 0
        self._last_stage_output_buffer = []
        self._last_stage_output_buffer_g0 = []
        self.num_iter_eval = 0
        self.num_iter_test = 0
        self.training = True

    @property
    def warmup_macrobatches(self):
        return self.num_warmup_macrobatches

    def add_transformed_zero_gradient(self, output):
        # this module can be optimized away
        def generate_zero_gradient(inputs):
            """for the nested list or tuple"""
            if isinstance(inputs, torch.Tensor):
                return torch.zeros_like(inputs)
            elif isinstance(inputs, (tuple, list)):
                result = []
                for item in inputs:
                    result.append(generate_zero_gradient(item))
                return result
            return None

        result = {}
        for k, v in output.items():
            if v is not None:
                zero_gradient = generate_zero_gradient(v)
                if zero_gradient is not None:
                    result[k] = zero_gradient
        self._last_stage_output_buffer_g0.append(result)

    def add_loss(self, losses):
        self._last_stage_output_buffer.append(losses)

    def warmup_iter_forward(self):
        raise NotImplementedError()

    def add_forward_output(self, output):
        """send the output of forward"""
        raise NotImplementedError()

    def warmup_iter_backward(self):
        raise NotImplementedError()

    def add_backward_output(self, output):
        """send the output of backward"""
        raise NotImplementedError()

    def __iter__(self):
        raise NotImplementedError()

    def add_output(self, output):
        """Alternate sending the forward and backward"""
        raise NotImplementedError()

    def _iter_backward(self, model_group: GroupClass):
        if self.stage == self.num_stages - 1 and model_group == GroupClass.G1:
            return self._last_stage_output_buffer.pop(0)
        return self._receive_backward_tensor(model_group)

    def _iter_forward(self, model_group: GroupClass):
        return self._receive_forward_tensor(model_group)

    def _send_forward_tensor(self, model_group: GroupClass, **kwargs):
        self._communication_helper.send(kwargs, DIRECTION.Forward, model_group)

    def _send_backward_tensor(self, model_group: GroupClass, **kwargs):
        self._communication_helper.send(kwargs, DIRECTION.Backward,
                                        model_group)

    def _receive_forward_tensor(self, model_group: GroupClass):
        if self.stage == 0 and model_group == GroupClass.G0:
            return next(self._data_loader)
        else:
            return self._communication_helper.recv(DIRECTION.Forward,
                                                   model_group)

    def _receive_backward_tensor(self, model_group: GroupClass):
        return self._communication_helper.recv(DIRECTION.Backward, model_group)

    def which_group(self, num_iters):
        if (num_iters // self.num_stages) % 2 == 0:
            return GroupClass.G0
        else:
            return GroupClass.G1

    def __len__(self):
        return self._dataset_length

    def set_eval_data_loader(self, data_loader):
        self._eval_data_loader = data_loader

    def turn_on_eval(self, times=2):
        self.training = False
        self.forward_iter = 0
        self._data_loader = iter(self._eval_data_loader)
        self.num_iter_eval = len(self._eval_data_loader) * times
        self._dataset_length = self.num_iter_eval

    def turn_on_train(self, times=2):
        self.training = True
        self.forward_iter = 0
        self.backward_iter = self.num_stages
        self.main_iter = 0
        self._data_loader = iter(self._train_data_loader)
        self._dataset_length = len(self._train_data_loader) * times


class UnifiedDataProvider(BasedUnifiedDataProvider):
    """
    """
    def __init__(self,
                 communication_helper: CommunicationHelper,
                 data_loader: DataLoader,
                 eval_data_loader: DataLoader = None,
                 test_data_loader: DataLoader = None):
        super(UnifiedDataProvider,
              self).__init__(communication_helper, data_loader,
                             eval_data_loader, test_data_loader)
        self.num_warmup_macrobatches = 2 * self.num_stages - 1 - self.stage

    def which_group_eval(self, num_iters):
        normal = (self.num_iter_eval //
                  (self.num_stages * 2)) * self.num_stages * 2
        if num_iters < normal:
            return self.which_group(num_iters)
        assert (self.num_iter_eval -
                normal) % 2 == 0, "error! num_iter_eval is not even"
        interval = (self.num_iter_eval - normal) // 2
        if num_iters - normal < interval:
            return GroupClass.G0
        return GroupClass.G1

    def eval_iter(self):
        for i in range(self.num_iter_eval):
            group_id = self.which_group_eval(self.forward_iter)
            yield self._iter_forward(group_id), group_id

    def pop_output(self, *args):
        if len(self._last_stage_output_buffer) > 0:
            outputs = self._last_stage_output_buffer.pop(0)
            return tuple(nested_detach_cpu(outputs[item]) for item in args)
        else:
            return tuple(None for _ in args)

    def warmup_iter_forward(self):
        for i in range(self.num_warmup_macrobatches):
            yield self._iter_forward(self.which_group(self.forward_iter))

    def warmup_iter_backward(self):
        for i in range(self.num_warmup_macrobatches):
            yield self._iter_backward(self.which_group(self.backward_iter))

    def __iter__(self):
        return self

    def __next__(self):
        if self.main_iter % 2 == 0:
            return self._iter_forward(self.which_group(self.forward_iter))
        else:
            return self._iter_backward(self.which_group(self.backward_iter))

    def add_forward_output(self, output):
        group_id = self.which_group(
            self.forward_iter) if self.training else self.which_group_eval(
                self.forward_iter)
        if self.stage == self.num_stages - 1 and group_id == GroupClass.G1:
            self.add_loss(output)
        else:
            self._send_forward_tensor(group_id, **output)

        self.forward_iter += 1

    def add_backward_output(self, output):
        self._send_backward_tensor(self.which_group(self.backward_iter),
                                   **output)
        self.backward_iter += 1

    def add_output(self, output):
        if self.main_iter % 2 == 0:
            self.add_forward_output(output)
        else:
            self.add_backward_output(output)
        self.main_iter += 1
