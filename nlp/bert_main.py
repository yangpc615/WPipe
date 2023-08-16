# coding=utf-8
""" Finetuning the library models for sequence classification on GLUE."""
import math
import dataclasses
import logging
import os
import sys
import json
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

import numpy as np

from transformers import (AutoConfig, AutoModelForSequenceClassification,
                          AutoTokenizer, EvalPrediction, GlueDataset,
                          BertConfig)

from transformers import GlueDataTrainingArguments as DataTrainingArguments
from transformers import (
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    glue_compute_metrics,
    glue_output_modes,
    glue_tasks_num_labels,
    set_seed,
)
from transformers.optimization import get_linear_schedule_with_warmup
from lib.runtimecontrol import WPipeTrainer
from nlp.bert_large_split import BertLargeUncastSplit
from lib.communicate import initialize
from lib.utils import get_fixed_schedule_without_warmup, count_parameters
from nlp.bert_large_split import BertLargeUncastSplit

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={
            "help":
            "Path to pretrained model or model identifier from huggingface.co/models"
        })
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help":
            "Pretrained config name or path if not the same as model_name"
        })
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help":
            "Pretrained tokenizer name or path if not the same as model_name"
        })
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help":
            "Where do you want to store the pretrained models downloaded from s3"
        })
    num_layers: Optional[int] = field(
        default=0,
        metadata={"help": "the number of layers,without pretrained"})


@dataclass
class RuntimeArguments:
    do_profile: bool = field(default=False, metadata={"help": "whether to record profile"})

    do_profile: bool = field(default=False,
                             metadata={"help": "whether to record profile"})

    do_recompute: bool = field(default=False,
                               metadata={"help": "whether to recompute"})

    do_broadcast: bool = field(
        default=False,
        metadata={"help": "whether to use broadcast to communicate"})

    do_throughput: bool = field(
        default=False, metadata={"help": "whether to compute throughput"})

    chunks: Optional[int] = field(
        default=0, metadata={"help": "the number of micro-batch"})

    network_config: Optional[str] = field(
        default=None, metadata={"help": "network configure for wpipe"})

    do_retrain: bool = field(default=False,
                             metadata={"help": "whether to continue training"})

    retrain_times: Optional[int] = field(
        default=1, metadata={"help": "the number of retrain"})

    is_scaler_fp16_p2p: bool = field(
        default=False, metadata={"help": "whether to communicate by fp16"})


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments,
                               TrainingArguments, RuntimeArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, runtime_args = parser.parse_args_into_dataclasses(
        )

    if (os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir) and training_args.do_train
            and not training_args.overwrite_output_dir):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
        if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    try:
        num_labels = glue_tasks_num_labels[data_args.task_name]
        output_mode = glue_output_modes[data_args.task_name]
    except KeyError:
        raise ValueError("Task not found: %s" % (data_args.task_name))

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    if model_args.num_layers <= 0:
        config = AutoConfig.from_pretrained(
            model_args.config_name
            if model_args.config_name else model_args.model_name_or_path,
            num_labels=num_labels,
            finetuning_task=data_args.task_name,
            cache_dir=model_args.cache_dir,
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
        )
    else:
        config = BertConfig(num_hidden_layers=model_args.num_layers)
        model = AutoModelForSequenceClassification.from_config(config)

    logger.info(
        f"The number of model parameters is {count_parameters(model)}!!!")
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name
        if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )
    # Get datasets
    train_dataset = (GlueDataset(data_args,
                                 tokenizer=tokenizer,
                                 cache_dir=model_args.cache_dir))
    eval_dataset = (GlueDataset(data_args,
                                tokenizer=tokenizer,
                                mode="dev",
                                cache_dir=model_args.cache_dir))
    test_dataset = (GlueDataset(data_args,
                                tokenizer=tokenizer,
                                mode="test",
                                cache_dir=model_args.cache_dir)
                    if training_args.do_predict else None)

    def build_compute_metrics_fn(task_name: str
                                 ) -> Callable[[EvalPrediction], Dict]:
        def compute_metrics_fn(p: EvalPrediction):
            preds = p.predictions[0] if isinstance(p.predictions,
                                                   tuple) else p.predictions
            if output_mode == "classification":
                preds = np.argmax(preds, axis=1)
            else:  # regression
                preds = np.squeeze(preds)
            return glue_compute_metrics(task_name, preds, p.label_ids)

        return compute_metrics_fn

    profile_dir_name = None
    if runtime_args.do_profile:
        profile_dir_name = "profile/wpipe"

    if runtime_args.network_config is None:
        # default network layout
        network_config = [
            ("stage0", {"model": [0, 4], "parallel": [0, 4]}),
            ("stage1", {"model": [1, 5], "parallel": [1, 5]}),
            ("stage2", {"model": [2, 6], "parallel": [2, 6]}),
            ("stage3", {"model": [3, 7], "parallel": [3, 7]}),
        ]
    else:
        with open(runtime_args.network_config, "r") as f:
            network_config = json.load(f)["network_config"]
    retrain_kwargs = {}
    if runtime_args.do_retrain:
        def get_lr_scheduler_wrapper(num_training_steps):
            def get_lr_scheduler(optimizer, *args, **kwargs):
                return get_fixed_schedule_without_warmup(optimizer)
            return get_lr_scheduler

        num_update_steps_per_epoch = len(train_dataset)
        num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
        max_steps = math.ceil(runtime_args.retrain_times *
        					  num_update_steps_per_epoch)
        retrain_kwargs = dict(
        optimizer_callbacks=(None,
        					 get_lr_scheduler_wrapper(max_steps)))
    trainer = WPipeTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=build_compute_metrics_fn(data_args.task_name),
        based_split_method=BertLargeUncastSplit,
        backend="gloo",
        network_config=network_config,
        do_recompute=runtime_args.do_recompute,
        do_broadcast=runtime_args.do_broadcast,
        do_throughput=runtime_args.do_throughput,
        profile_dir_name=profile_dir_name,
        is_scaler_fp16_p2p=runtime_args.is_scaler_fp16_p2p,
        **retrain_kwargs
    )

    # Training
    eval_result_list = []
    if training_args.do_train:
        trainer.train()
        trainer.save_model()
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_master():
            tokenizer.save_pretrained(training_args.output_dir)
        if runtime_args.do_retrain:
            for i in range(runtime_args.retrain_times - 1):
                logger.info("*** Evaluate ***")

                # Loop to handle MNLI double evaluation (matched, mis-matched)
                eval_datasets = [eval_dataset]
                if data_args.task_name == "mnli":
                    mnli_mm_data_args = dataclasses.replace(
                        data_args, task_name="mnli-mm")
                    eval_datasets.append(
                        GlueDataset(mnli_mm_data_args,
                                    tokenizer=tokenizer,
                                    mode="dev",
                                    cache_dir=model_args.cache_dir))

                cur_results = []
                for eval_dataset in eval_datasets:
                    trainer.compute_metrics = build_compute_metrics_fn(
                        eval_dataset.args.task_name)
                    eval_result = trainer.evaluate(eval_dataset=eval_dataset)
                    if trainer.is_world_master():
                        for key, value in eval_result.items():
                            logger.info("  %s = %s", key, value)
                        cur_results.append(eval_result)
                        eval_result_list.append(cur_results)
                trainer.set_seed(i + 1)
                trainer.train(retrain=True)

    # Evaluation
    eval_results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        eval_datasets = [eval_dataset]
        if data_args.task_name == "mnli":
            mnli_mm_data_args = dataclasses.replace(data_args,
                                                    task_name="mnli-mm")
            eval_datasets.append(
                GlueDataset(mnli_mm_data_args,
                            tokenizer=tokenizer,
                            mode="dev",
                            cache_dir=model_args.cache_dir))

        last_result = []
        for eval_dataset in eval_datasets:
            trainer.compute_metrics = build_compute_metrics_fn(
                eval_dataset.args.task_name)
            eval_result = trainer.evaluate(eval_dataset=eval_dataset)
            last_result.append(eval_result)

            output_eval_file = os.path.join(
                training_args.output_dir,
                f"eval_results_{eval_dataset.args.task_name}.txt")
            if trainer.is_world_master():
                if not os.path.exists(training_args.output_dir):
                    os.makedirs(training_args.output_dir)
                with open(output_eval_file, "w") as writer:
                    logger.setLevel(logging.INFO)
                    logger.info("***** Eval results {} *****".format(
                        eval_dataset.args.task_name))
                    for key, value in eval_result.items():
                        logger.info("  %s = %s", key, value)
                        writer.write("%s = %s\n" % (key, value))

                eval_results.update(eval_result)
        eval_result_list.append(last_result)
        if trainer.is_world_master():
            with open(
                    os.path.join(
                        training_args.output_dir,
                        f"eval_all_results_{eval_dataset.args.task_name}.json"
                    ), 'w') as fp:
                json.dump({"eval_results": eval_result_list}, fp)

    if training_args.do_predict:
        logging.info("*** Test ***")
        test_datasets = [test_dataset]
        if data_args.task_name == "mnli":
            mnli_mm_data_args = dataclasses.replace(data_args,
                                                    task_name="mnli-mm")
            test_datasets.append(
                GlueDataset(mnli_mm_data_args,
                            tokenizer=tokenizer,
                            mode="test",
                            cache_dir=model_args.cache_dir))

        for test_dataset in test_datasets:
            predictions = trainer.predict(
                test_dataset=test_dataset).predictions
            if output_mode == "classification":
                predictions = np.argmax(predictions, axis=1)

            output_test_file = os.path.join(
                training_args.output_dir,
                f"test_results_{test_dataset.args.task_name}.txt")
            if trainer.is_world_master():
                with open(output_test_file, "w") as writer:
                    logger.info("***** Test results {} *****".format(
                        test_dataset.args.task_name))
                    writer.write("index\tprediction\n")
                    for index, item in enumerate(predictions):
                        if output_mode == "regression":
                            writer.write("%d\t%3.3f\n" % (index, item))
                        else:
                            item = test_dataset.get_labels()[item]
                            writer.write("%d\t%s\n" % (index, item))
    return eval_results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
