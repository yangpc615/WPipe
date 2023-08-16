import os
import argparse
import subprocess
import sys
import signal
import time
import torch.distributed as dist


parser = argparse.ArgumentParser(description="running")
parser.add_argument('-d',
                    '--depth',
                    type=int,
                    help="the depth of model parallelism")
parser.add_argument('-n',
                    '--num_data_parallel',
                    type=int,
                    help='the number of data parallel')
parser.add_argument('-a', '--arch', type=str, help="the name of model")
parser.add_argument('--num_layers', type=int, help="number of bert")
parser.add_argument("--method", type=str, help="the pipeline parallel method")
parser.add_argument("--nnodes",
                    type=int,
                    default=8,
                    help="The number of nodes to use for distributed "
                    "training")
parser.add_argument("--node_rank",
                    type=int,
                    default=0,
                    help="The rank of the node for multi-node distributed "
                    "training")
parser.add_argument("--nproc_per_node",
                    type=int,
                    default=8,
                    help="The number of processes to launch on each node, "
                    "for GPU training, this is recommended to be set "
                    "to the number of GPUs in your system so that "
                    "each process can be bound to a single GPU.")
parser.add_argument("--master_addr",
                    default="127.0.0.1",
                    type=str,
                    help="Master node (rank 0)'s address, should be either "
                    "the IP address or the hostname of node 0, for "
                    "single node multi-proc training, the "
                    "--master_addr can simply be 127.0.0.1")
parser.add_argument("--master_port",
                    default=29500,
                    type=int,
                    help="Master node (rank 0)'s free port that needs to "
                    "be used for communication during distributed "
                    "training")
parser.add_argument("--data_dir",
                    type=str,
                    default="/home/myself/wpipe",
                    help="the data directory")
parser.add_argument("-b", "--batch_size", type=int, default=32, help="the batch size")
parser.add_argument("--do_recompute",
                    action='store_true',
                    help="whether to recompute")
parser.add_argument("--do_batch",
                    action='store_true',
                    help="whether to run by batch")
parser.add_argument("--task", choices=["cv", "nlp"], help="class of task")
parser.add_argument("--do_multi",
                    action='store_true',
                    help="whether to train by multi-machine")

TASK_ID = 0


def single_running(args):
    is_recompute = ""
    if args.do_recompute:
        is_recompute = "--do_recompute"
    cv_config_name = "network_conf/throughput{}x{}cv.json".format(
        args.depth, args.num_data_parallel)
    running_cv = f"python -m torch.distributed.launch --nproc_per_node  \
      {args.nproc_per_node}  --master_port {args.master_port} --master_addr \
      {args.master_addr} --nnodes {args.nnodes} --node_rank {args.node_rank} \
       cv/resnext_main.py --data_dir {args.data_dir} -a {args.arch} --dist-backend nccl  -j  20  -b \
      {args.batch_size} --lr 0.01   --dataset_name  oxfordflowers102  --num_train_epochs  95  \
      --output_dir  throughput_cv  --seed 42 --warmup_ratio 0.05  --run_method {args.method} \
      --network_config {cv_config_name} {is_recompute}  --do_throughput --is_scaler_fp16_p2p"

    nlp_config_name = "network_conf/throughput{}x{}nlp.json".format(args.depth,
                                                  args.num_data_parallel)
    pipeline_method = ""
    running_nlp = f"python -m torch.distributed.launch --nproc_per_node  \
      {args.nproc_per_node}  --master_port {args.master_port} --master_addr \
      {args.master_addr} --nnodes {args.nnodes} --node_rank {args.node_rank} \
      nlp/bert_main.py --model_name_or_path bert-base-uncased --task_name QQP --do_train \
      --data_dir {os.path.join(args.data_dir, 'QQP')} \
      --max_seq_length 128 \
      --num_layers {args.num_layers} \
      --per_device_train_batch_size {args.batch_size} \
      --learning_rate 16e-5 \
      --warmup_steps  1000 \
      --num_train_epochs 15.0 \
      --seed 42 \
      --output_dir throughput_nlp  --overwrite_output_dir --dataloader_drop_last \
      {pipeline_method} --network_config {nlp_config_name} \
      {is_recompute} --do_throughput "

    output_dir = None
    outputs = None
    if args.arch.lower().startswith("resnext"):
        cmd = running_cv
        output_dir = "throughput_cv"
        outputs = [
            args.arch, str(args.batch_size), args.method, cv_config_name,
            is_recompute
        ]
    elif args.arch.lower().startswith("bert"):
        cmd = running_nlp
        output_dir = "throughput_nlp"
        outputs = [
            str(args.num_layers), str(args.batch_size), pipeline_method, nlp_config_name,
            is_recompute
        ]
    else:
        raise ValueError(f"The model {args.arch} is not supported!")

    run_cmd(cmd, args.node_rank, output_dir, outputs)


def run_cmd(cmd, node_rank, output_dir, outputs, task_id=0):
    sig_names = {2: "SIGINT", 15: "SIGTERM"}
    last_return_code = None

    def sigkill_handler(signum, frame):
        for process in processes:
            print(f"Killing subprocess {process.pid}")
            try:
                process.kill()
            except Exception as e:
                pass
        if last_return_code is not None:
            raise subprocess.CalledProcessError(returncode=last_return_code, cmd=cmd)
        if signum in sig_names:
            print(f"Main process received {sig_names[signum]}, exiting")
        sys.exit(1)

    # pass SIGINT/SIGTERM to children if the parent is being terminated
    signal.signal(signal.SIGINT, sigkill_handler)
    signal.signal(signal.SIGTERM, sigkill_handler)

    current_env = os.environ.copy()
    if node_rank == 0:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with open(os.path.join(output_dir, "throughput.txt"), "a+") as f:
            f.writelines("\n" + ",".join(outputs))

    process = subprocess.Popen(cmd, env=current_env, shell=True)
    processes = [process]
    try:
        alive_processes = set(processes)
        while len(alive_processes):
            finished_processes = []
            for process in alive_processes:
                if process.poll() is None:
                    # the process is still running
                    continue
                else:
                    if process.returncode != 0:
                        last_return_code = process.returncode  # for sigkill_handler
                        sigkill_handler(signal.SIGTERM, None)  # not coming back
                    else:
                        # exited cleanly
                        finished_processes.append(process)
            alive_processes = set(alive_processes) - set(finished_processes)

            time.sleep(1)
    except Exception:
        killer_cmd = "ps -ef | grep local_rank | awk '{print $2}' | xargs kill"
        process = subprocess.Popen(killer_cmd, env=current_env, shell=True)
        process.wait()
        if node_rank == 0:
            with open(os.path.join(output_dir, "throughput.txt"), "a+") as f:
                f.writelines(",\terror!!!!\n")
    finally:
        if node_rank == 0:
            with open(os.path.join(output_dir, "task_id.txt"), "w") as f:
                f.writelines(f"{task_id}")



# CV_models = [
#     "resnext50_32x4d", "resnext101_32x4d", "resnext152_32x4d",
#     "resnext200_32x4d", "resnext302_32x4d"
# ]
# single machine
# NLP_models = [24, 48, 96, 192, 384]
# CV_batch = [2**i for i in (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)]
# NLP_batch = [2**i for i in (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)]
# RUN_method = ["apipeplus"]
# RUN_method_nlp = [""]
# Network_conf = list(reversed([(2, 32), (4, 16), (8, 8), (16, 4), (32, 2), (64, 1)]))
# Network_conf_single = [(8, 1), (4, 2), (2, 4)]
# IS_recompute = ["--do_recompute", ""]

CV_models = [
   "resnext500_32x4d", "resnext800_32x4d"
]
NLP_models = list(reversed([384, 768]))
CV_batch = [2**i for i in (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)]
NLP_batch = [2**i for i in (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)]
RUN_method = ["wpipe"]
RUN_method_nlp = [""]
Network_conf = list(reversed([(8, 4), (16, 2), (32, 1)]))
Network_conf_single = [(8, 1), (4, 2), (2, 4)]
IS_recompute = ["--do_recompute", ""]


def generate_args(kind: str, is_single):
    config_args = []
    conf_network = Network_conf_single if is_single else Network_conf
    if "cv" in kind:
        for model in CV_models:
            for recompute in IS_recompute:
                for batch_size in CV_batch:
                    for run_method in RUN_method:
                        for network in conf_network:
                            config_args.append(
                                (model, str(batch_size), run_method,
                                 "network_conf/throughput{}x{}cv.json".format(
                                     *network), recompute))
        return config_args
    else:
        for num_layers in NLP_models:
            for recompute in IS_recompute:
                for batch_size in NLP_batch:
                    for run_method in RUN_method_nlp:
                        for network in conf_network:
                            config_args.append(
                                (str(num_layers), str(batch_size), run_method,
                                 "network_conf/throughput{}x{}nlp.json".format(
                                     *network), recompute))
        return config_args


def batch_running(args):
    distributed_head = f"python -m torch.distributed.launch --nproc_per_node  \
                        {args.nproc_per_node}  --master_port {args.master_port} --master_addr \
                        {args.master_addr} --nnodes {args.nnodes} --node_rank {args.node_rank}  "

    running_cv = " resnext_main.py {}  -a {} --dist-backend nccl  \
      -j  20  -b  {} --lr 0.01   --dataset_name  oxfordflowers102  --num_train_epochs 995  \
      --output_dir  throughput_cv  --seed 42 --warmup_ratio 0.05  --run_method {} \
      --network_config {} {}  --do_throughput --is_scaler_fp16_p2p"

    running_nlp = " bert_main.py --model_name_or_path bert-base-uncased --task_name QQP --do_train \
      --data_dir {} --max_seq_length 128 --num_layers {} --per_device_train_batch_size {} \
      --learning_rate 16e-5 --warmup_steps  1000 --num_train_epochs 15.0 --seed 42 \
      --output_dir throughput_nlp  --overwrite_output_dir --dataloader_drop_last \
      {}  --network_config {}  {} --do_throughput"

    method = "tcp://" + args.master_addr + ":23654"
    dist.init_process_group(backend='gloo', init_method=method, world_size=args.nnodes, rank=args.node_rank)
    global TASK_ID
    output_dir = "throughput_cv" if args.task == "cv" else "throughput_nlp"
    with open(os.path.join(output_dir, "task_id.txt"), "r") as f:
        task_id = int(f.readline())
        TASK_ID = task_id
    if "cv" == args.task:
        config_args = generate_args("cv", not args.do_multi)[TASK_ID:]
        cv_dir = os.path.join(args.data_dir, '')
        for i, config in enumerate(config_args):
            running_cmd = distributed_head + running_cv.format(*(
                (cv_dir, ) + config))
            run_cmd([running_cmd], args.node_rank, "throughput_cv", config, i + TASK_ID + 1)
            dist.barrier()
    elif "nlp" == args.task:
        config_args = generate_args("nlp", not args.do_multi)[TASK_ID:]
        nlp_dir = os.path.join(args.data_dir, 'QQP')
        for i, config in enumerate(config_args):
            running_cmd = distributed_head + running_nlp.format(*(
                (nlp_dir, ) + config))
            run_cmd([running_cmd], args.node_rank, "throughput_nlp", config, i + TASK_ID + 1)
            dist.barrier()
    else:
        raise ValueError(f"The model {args.arch} is not supported!")


if __name__ == "__main__":
    args = parser.parse_args()
    if args.do_batch:
        batch_running(args)
    else:
        single_running(args)
