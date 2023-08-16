import os
import logging
import collections
import math
import numpy as np
import torch
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Text
from tqdm import tqdm
from tqdm.auto import trange
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.dataset import Dataset
from transformers import Trainer, TrainingArguments, set_seed
from transformers.data.data_collator import (
    default_data_collator,
    DataCollator,
    DataCollatorWithPadding
)
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from transformers.trainer_callback import TrainerCallback
from transformers.modeling_utils import PreTrainedModel
from transformers.trainer_utils import EvalPrediction, PredictionOutput
from transformers.trainer_pt_utils import SequentialDistributedSampler

from lib.unifieddatainterface import UnifiedDataProvider, GroupClass
from lib.communicate import CommunicationHelper, COMMETHOD
from lib.layout import (
    GlobalDataParallelLayout, 
    InterfaceParam, 
    GroupClass, 
    DataModelParallelLayout
)
from lib.generatecallableunit import GenerateCallableUnit
from lib.optimizer import DoubleGroupsOptimizer
from lib.callableunit import GroupIndex
from lib.utils import (
    get_samper_args, 
    LossManager, 
    count_tensor_by_size, 
    MultiEpochDataLoader,
    ProfileHelper, 
    DistributedProgressBar, 
    EvalDecorator, 
    distributed_concat, 
    nested_concat,
    ThroughPutCalculate,
    clip_input,
    CalculateMemory
)

logger = logging.getLogger(__name__)
ClassToIndex = {GroupClass.G0: GroupIndex.G0, GroupClass.G1: GroupIndex.G1}


class RuntimeControl(object):
    """
    runtime control

    args:
    model: the raw model
    based_split_method: used to split model
    dataloader: data loader
    netword_config: stage data parallel and model parallel configure
    backend: the backend communication protocol
    local_rank: the local rank
    is_broadcast: if it's True, the communication is implemented by broadcast , otherwise, by Network
    is_recomputed: control whether the G1 to recompute
    num_iterations: the number of iterations for each rank
    optim_name: the names of optimizers in torch.optim
    optim_args: the parameters of the optimizer
    """
    def __init__(self,
                 model,
                 based_split_method,
                 dataloader,
                 eval_dataloader,
                 network_config,
                 backend,
                 local_rank,
                 is_broadcast,
                 is_recomputed,
                 num_iterations,
                 num_epochs,
                 is_scaler_fp16_p2p=True,
                 output_dir=None,
                 profile_dir_name=None,
                 num_loss_sample=120):
        self._interface_param = InterfaceParam()
        self._layout = self._create_layout(network_config=network_config,
                                           backend=backend,
                                           local_rank=local_rank,
                                           is_broadcast=is_broadcast)
        self.num_warmup_size = self._layout.num_stages - self._layout.stage - 1
        self.stage = self._layout.stage
        self.is_recomputed = is_recomputed
        self.local_rank = local_rank
        num_iterations = (num_iterations //
                          self._layout.num_stages) * self._layout.num_stages
        self.num_iterations = num_iterations * 2
        self._generate_callable_tool = GenerateCallableUnit(
            model, based_split_method, is_recomputed, local_rank,
            (self._layout.get_data_parallel_group(), ) * 2, self.stage)
        self._data_loader = MultiEpochDataLoader(dataloader, num_epochs)
        # must be before self.init_interface()
        self.num_partitions = 2 * self._layout.num_stages
        self.init_interface()
        # this must be after the self.init_interface()
        if hasattr(self._layout, "init_model_parallel_group"):
            self._layout.init_model_parallel_group(self._interface_param)
            logger.info("building connection in the same group starts!")
            self._layout.build_connection_in_same_group()
            logger.info("building connection in the same group finished!")

        self._communication_helper = CommunicationHelper(
            self._layout, self._interface_param,
            self._calculate_num_iterations(num_iterations),
            is_fp16=is_scaler_fp16_p2p)
        self._communication_helper.initialize(False)
        self._unified_data_provider = UnifiedDataProvider(
            self._communication_helper, self._data_loader, eval_dataloader)
        self._callable_unit = self._generate_callable_tool.generate_callable_unit(
            method=self.num_partitions)
        self._loss_manager = LossManager(num_iterations // num_loss_sample)
        self._pbar = None
        self._throughput = None
        self._calculate_memory = None
        self.output_dir = output_dir if output_dir is not None else ""
        self.profile_helper = None
        if profile_dir_name is not None:
            self.profile_helper = ProfileHelper(
                profile_dir_name=profile_dir_name,
                profile_step_num=12,
                distributed=True,
                stage=self.stage)
            self.profile_helper.set_delay([0, 0, 0, 0])

    def set_seed(self, seed):
        self._data_loader.set_seed(seed)

    def retrain(self, times=2):
        self._pbar = None
        self._communication_helper.initialize(False)
        self._unified_data_provider.turn_on_train(times=times)
        if self.__class__.__name__ == "RuntimeControl":
            self._callable_unit.reset()

    @property
    def layout(self):
        return self._layout

    def _create_layout(self, **kwargs):
        """
        create layout object based on kwargs
        """
        if kwargs["is_broadcast"] or kwargs["backend"] == 'nccl':
            return DataModelParallelLayout(**kwargs)
        else:
            return GlobalDataParallelLayout(**kwargs)

    def get_current_model(self):
        return self._callable_unit

    def set_optimizers_scheduler(self,
                                 optimizer_g0,
                                 optimizer_g1,
                                 lr_scheduler_g0=None,
                                 lr_scheduler_g1=None):
        self._double_group_optimizer = DoubleGroupsOptimizer(
            optimizer_g0,
            optimizer_g1,
            lr_scheduler_g0,
            lr_scheduler_g1,
            self._callable_unit,
            gradient_acc_interval=self._layout.num_stages)

    def cuda(self):
        self._callable_unit.cuda(self.local_rank)

    def init_interface(self):
        partitions = self._generate_callable_tool.split_model(
            nonmean=False, method=self.num_partitions)
        inputs = next(iter(self._data_loader))
        clip_input(inputs, batch_size=1)
        for i, part in enumerate(partitions):
            group = GroupClass.G0 if i < len(partitions) / 2 else GroupClass.G1
            self._interface_param.add_stage_input_keys(inputs, group)
            outputs = part(**inputs)
            self._interface_param.add_stage_output_keys(outputs, group)
            inputs = outputs
        logger.info("initializing interface finished!")

    def print_loss(self, outputs, par_len, optimizer=None):
        group = self._layout.get_data_parallel_group()
        cur_rank = torch.distributed.get_rank(group=group)
        data_parallel_size = torch.distributed.get_world_size(group=group)
        if self._pbar is None:
            self._pbar = DistributedProgressBar(total=par_len,
                                                rank=self._layout.num_stages -
                                                1)
        if self._throughput is None:
            self._throughput = ThroughPutCalculate(
                64, data_parallel_size, self._data_loader.batch_size)
            if self.stage == self._layout.num_stages - 1 and cur_rank == 0:
                # start for init
                self._throughput.update(1)
        if self._calculate_memory is None:
            self._calculate_memory = CalculateMemory()
        self._calculate_memory.calculate_memory()
        self._pbar.update(1)
        if "loss" in outputs or "losses" in outputs:
            key = "loss" if "loss" in outputs else "losses"
            loss = outputs[key].detach().clone()
            torch.distributed.all_reduce(loss, group=group)
            world_size = torch.distributed.get_world_size(group=group)
            if cur_rank == 0:
                lr = optimizer.get_lr() if optimizer else ()
                self._loss_manager.add(loss / world_size, *lr)
                self._throughput.update(1)

    def warmup_run_forward(self):
        logger.info("forward warmup running starts!")
        self._callable_unit.train()
        for inputs in self._unified_data_provider.warmup_iter_forward():
            outputs = self._callable_unit(**inputs)
            self._unified_data_provider.add_forward_output(outputs)
        logger.info("forward warmup running finished!")

    def run_forward_backward(self):
        logger.info("forward backward running starts!")
        self._callable_unit.train()
        num_iters = self.num_iterations - self._unified_data_provider.warmup_macrobatches
        for i in range(num_iters):
            if self.profile_helper:
                self.profile_helper.start_check(i)
            forward_inputs = next(self._unified_data_provider)
            forward_outputs = self._callable_unit(**forward_inputs)
            self._unified_data_provider.add_output(forward_outputs)

            self.print_loss(forward_outputs, num_iters,
                            self._double_group_optimizer)
            # backward_inputs = next(self._unified_data_provider)
            data_provider = self._unified_data_provider
            # self._double_group_optimizer.zero_grad()
            backward_outputs = self._callable_unit.backward(
                data_provider=data_provider)
            self._unified_data_provider.add_output(backward_outputs)
            self._double_group_optimizer.step()
            if self.profile_helper:
                self.profile_helper.end_check(i, 0)
        self._pbar.close()
        logger.info("forward backward running finished!")

    def warmup_run_backward(self):
        logger.info("backward warmup running starts!")
        self._callable_unit.train()
        for inputs in self._unified_data_provider.warmup_iter_backward():
            outputs = self._callable_unit.backward(
                register_hook_for_reducer=True, **inputs)
            self._unified_data_provider.add_backward_output(outputs)
        self._loss_manager.save_data(
            os.path.join(self.output_dir, "losses.csv"))
        self._throughput.save_data(
            os.path.join(self.output_dir, "throughput.txt"))
        self._calculate_memory.save_data(
            os.path.join(self.output_dir, "throughput.txt"))
        logger.info("backward warmup running finished!")

    def _calculate_num_iterations(self, num_iterations):
        return num_iterations

    @EvalDecorator
    def run_eval(self, data_loader=None, names=None):
        logger.info("eval running starts!")
        self._callable_unit.eval()
        results = []
        if data_loader is not None:
            self._unified_data_provider.set_eval_data_loader(data_loader)
        self._unified_data_provider.turn_on_eval()
        dataset_length = len(self._unified_data_provider)
        self._communication_helper.initialize(True,
                                              eval_num_iters=dataset_length //
                                              2)
        self._pbar = DistributedProgressBar(dataset_length, 0)
        for inputs, group_id in self._unified_data_provider.eval_iter():
            outputs = self._callable_unit(group_id=ClassToIndex[group_id],
                                          **inputs)
            self._unified_data_provider.add_forward_output(outputs)
            outputs = self._unified_data_provider.pop_output(*names)
            self._pbar.update(1)
            if any(outputs):
                results.append(outputs)
        self._pbar.close()
        logger.info("eval running finished!")
        return results

class WPipeTrainer(Trainer):
    """
    Trainer is a simple but feature-complete training and eval loop for WPipe,
    """
    def __init__(
            self,
            model: Union[PreTrainedModel, torch.nn.Module] = None,
            args: TrainingArguments = None,
            data_collator: Optional[DataCollator] = None,
            train_dataset: Optional[Dataset] = None,
            eval_dataset: Optional[Dataset] = None,
            tokenizer: Optional["PreTrainedTokenizerBase"] = None,
            model_init: Callable[[], PreTrainedModel] = None,
            compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
            callbacks: Optional[List[TrainerCallback]] = (None, ),
            optimizer_callbacks: Tuple[Callable, Callable] = (None, None),
            based_split_method: Callable = None,
            network_config: Dict = None,
            backend: Text = None,
            do_pipedream_2bw=False,
            do_recompute=False,
            do_broadcast=True,
            profile_dir_name=None,
            **kwargs,
    ):
        if args is None:
            logger.info(
                "No `TrainingArguments` passed, using the current path as `output_dir`."
            )
            args = TrainingArguments("tmp_trainer")
        self.args = args
        # Seed must be set before instantiating the model when using model
        set_seed(self.args.seed)
        torch.cuda.set_device(self.args.local_rank)
        assert (
            model is not None or model_init is not None
        ), "You must provide a model to use `Trainer`, either by using the `model` argument \
           or the `model_init` argument."

        self.do_throughput = kwargs["do_throughput"]
        self.is_scaler_fp16_p2p = kwargs["is_scaler_fp16_p2p"]
        self.model_init = model_init
        if model is None and model_init is not None:
            model = self.call_model_init()
        default_collator = default_data_collator if tokenizer is None else DataCollatorWithPadding(
            tokenizer)
        self.data_collator = data_collator if data_collator is not None else default_collator
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer

        self.compute_metrics = compute_metrics
        self.optimizer, self.lr_scheduler = None, None
        self._get_optimizer, self._get_lr_scheduler = optimizer_callbacks
        self.model = model
        self.based_split_method = based_split_method
        self.network_config = network_config
        self.backend = backend
        self.do_recompute = do_recompute
        self.do_broadcast = do_broadcast
        self.profile_dir_name = profile_dir_name
        self.init_runtimecontrol()
        self._collection_data = callbacks[0]

    def _get_train_sampler(self) -> Optional[torch.utils.data.sampler.Sampler]:
        if not isinstance(self.train_dataset, collections.abc.Sized):
            return None
        else:
            return (RandomSampler(self.train_dataset)
                    if self.args.local_rank == -1 else DistributedSampler(*(
                        (self.train_dataset, ) +
                        get_samper_args(self.network_config))))

    def init_runtimecontrol(self):
        train_dataloader = self.get_train_dataloader()
        eval_dataloader = self.get_eval_dataloader(self.eval_dataset)
        train_dataset_is_sized = isinstance(self.train_dataset,
                                            collections.abc.Sized)
        if train_dataset_is_sized:
            num_update_steps_per_epoch = len(train_dataloader)
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            max_steps = math.ceil(self.args.num_train_epochs *
                                  num_update_steps_per_epoch)
            num_train_epochs = math.ceil(self.args.num_train_epochs)

        num_iterations = max_steps
        if self.do_throughput:
            num_iterations = 128
        self.runtime_control = RuntimeControl(
            self.model,
            self.based_split_method,
            train_dataloader,
            eval_dataloader,
            self.network_config,
            self.backend,
            self.args.local_rank,
            self.do_broadcast,
            self.do_recompute,
            num_iterations,
            num_train_epochs,
            is_scaler_fp16_p2p=self.is_scaler_fp16_p2p,
            output_dir=self.args.output_dir,
            profile_dir_name=self.profile_dir_name,
            num_loss_sample=min(120, num_iterations // 2))
        self.model = self.runtime_control.get_current_model()
        self.create_optimizer_and_scheduler(
            max_steps // self.runtime_control.layout.num_stages)
        self.runtime_control.set_optimizers_scheduler(*(self.optimizer +
                                                        self.lr_scheduler))

    def set_seed(self, seed):
        self.runtime_control.set_seed(seed)

    def train(self, retrain: Optional[bool] = False):
        # runtime
        if self.do_throughput:
            logger.setLevel(logging.ERROR)
        if retrain:
            times = 2
            self.runtime_control.retrain(times)
        self.runtime_control.warmup_run_forward()
        self.runtime_control.run_forward_backward()
        self.runtime_control.warmup_run_backward()

    def _get_eval_sampler(self, eval_dataset: Dataset
                          ) -> Optional[torch.utils.data.sampler.Sampler]:
        if self.args.local_rank != -1:
            return SequentialDistributedSampler(*(
                (eval_dataset, ) + get_samper_args(self.network_config)))
        else:
            return SequentialSampler(eval_dataset)

    def is_world_master(self) -> bool:
        group = self.runtime_control.layout.get_data_parallel_group()
        rank = torch.distributed.get_rank(group=group)
        return self.runtime_control.layout.is_output_stage() and rank == 0

    def evaluate(self,
                 eval_dataset: Optional[Dataset] = None) -> Dict[str, float]:
        """
        Run evaluation and returns metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are
        task-dependent (pass it to the init :obj:`compute_metrics` argument).

        You can also subclass and override this method to inject custom behavior.

        Args:
            eval_dataset (:obj:`Dataset`, `optional`):
                Pass a dataset if you wish to override :obj:`self.eval_dataset`. If it is an :obj:`datasets.Dataset`,
                columns not accepted by the ``model.forward()`` method are automatically removed. It must implement
                the :obj:`__len__` method.

        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions.
        """
        if eval_dataset is not None and not isinstance(eval_dataset,
                                                       collections.abc.Sized):
            raise ValueError("eval_dataset must implement __len__")

        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        result = self.runtime_control.run_eval(data_loader=eval_dataloader,
                                               names=("loss", "output",
                                                      "labels"))
        if len(result) > 0:
            losses, outputs, labels = self.collection_data(result)

            group = self.runtime_control.layout.get_data_parallel_group()
            device_id = self.args.local_rank
            eval_loss = distributed_concat(losses, group, device_id)
            preds = distributed_concat(outputs, group, device_id)
            label_ids = distributed_concat(labels, group, device_id)

            if preds is not None:
                preds = preds.cpu().numpy()
            if label_ids is not None:
                label_ids = label_ids.cpu().numpy()

            if self.compute_metrics is not None and preds is not None and label_ids is not None:
                metrics = self.compute_metrics(
                    EvalPrediction(predictions=preds, label_ids=label_ids))
            else:
                metrics = {}

            if eval_loss is not None:
                metrics["eval_loss"] = eval_loss.mean().item()

            # Prefix all keys with eval_
            for key in list(metrics.keys()):
                if not key.startswith("eval_"):
                    metrics[f"eval_{key}"] = metrics.pop(key)

            output = PredictionOutput(predictions=preds,
                                      label_ids=label_ids,
                                      metrics=metrics)

            # self.log(output.metrics)
            return output.metrics

    def collection_data(self, results):
        if self._collection_data is not None:
            return self._collection_data(results)
        losses = torch.stack([item[0] for item in results])
        outputs = torch.cat([item[1][0] for item in results])
        labels = torch.cat([item[2] for item in results])

        return losses, outputs, labels

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """
        Setup the optimizer and the learning rate scheduler.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through :obj:`optimizers`, or subclass and override this method in a subclass.
        """
        def get_optimizer(model):
            no_decay = ["bias", "LayerNorm.weight"]

            optimizer_parameters = [
                {
                    "params": [
                        p for n, p in model.named_parameters()
                        if not any(nd in n for nd in no_decay)
                    ],
                    "weight_decay":
                    self.args.weight_decay,
                },
                {
                    "params": [
                        p for n, p in model.named_parameters()
                        if any(nd in n for nd in no_decay)
                    ],
                    "weight_decay":
                    0.0,
                },
            ]
            optimizer = AdamW(
                optimizer_parameters,
                lr=self.args.learning_rate,
                betas=(self.args.adam_beta1, self.args.adam_beta2),
                eps=self.args.adam_epsilon,
            )
            return optimizer

        def get_lr_scheduler(optimizer):
            lr_scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.args.warmup_steps,
                num_training_steps=num_training_steps)
            return lr_scheduler

        if self._get_optimizer is None:
            self._get_optimizer = get_optimizer
        if self._get_lr_scheduler is None:
            self._get_lr_scheduler = get_lr_scheduler

        if self.optimizer is None:
            self.optimizer = [None] * 2
            self.optimizer[0] = self._get_optimizer(
                self.model.group_models[GroupIndex.G0].module)
            self.optimizer[1] = self._get_optimizer(
                self.model.group_models[GroupIndex.G1].module)

        if self.lr_scheduler is None:
            self.lr_scheduler = [None] * 2
            self.lr_scheduler[0] = self._get_lr_scheduler(self.optimizer[0])
            self.lr_scheduler[1] = self._get_lr_scheduler(self.optimizer[1])
