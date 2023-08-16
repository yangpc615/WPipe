import argparse
import logging
import os
import random
import shutil
import time
import warnings
import json
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.optim.lr_scheduler import LambdaLR
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torchvision.models.resnet import model_urls, ResNet, Bottleneck
from torchvision.models.utils import load_state_dict_from_url
from lib.utils import (
    DistributedProgressBar, 
    LossManager, 
    BatchNorm2dClone, 
    ThroughPutCalculate,
	get_fixed_schedule_without_warmup,
	CalculateMemory
)
from lib.runtimecontrol import WPipeTrainer
from cv.dataset_provider import get_dataloader_func
from cv.resnext_split import ResnextSplit
from cv.callbacks import resnext_collate

logger = logging.getLogger(__name__)

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data_dir', metavar='DIR', help='path to dataset')
parser.add_argument('-a',
                    '--arch',
                    metavar='ARCH',
                    default='resnet18',
                    help='model architecture: ' + ' | '.join(model_names) +
                    ' (default: resnet18)')
parser.add_argument('-j',
                    '--dataloader_num_workers',
                    default=4,
                    type=int,
                    metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--num_train_epochs',
                    default=95,
                    type=int,
                    metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch',
                    default=0,
                    type=int,
                    metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument(
    '-b',
    '--train-batch-size',
    default=32,
    type=int,
    metavar='N',
    help='mini-batch size (default: 32x8 = 256), this is the total '
    'batch size of all GPUs on the current node when '
    'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--eval-batch-size', default=32, type=int)
parser.add_argument('--lr',
                    '--learning-rate',
                    default=0.1,
                    type=float,
                    metavar='LR',
                    help='initial learning rate',
                    dest='lr')
parser.add_argument('--momentum',
                    default=0.9,
                    type=float,
                    metavar='M',
                    help='momentum')
parser.add_argument('--wd',
                    '--weight-decay',
                    default=1e-4,
                    type=float,
                    metavar='W',
                    help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-eval_prediction',
                    '--print-freq',
                    default=1000,
                    type=int,
                    metavar='N',
                    help='print frequency (default: 10)')
parser.add_argument('--resume',
                    default='',
                    type=str,
                    metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e',
                    '--evaluate',
                    dest='evaluate',
                    action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained',
                    dest='pretrained',
                    action='store_true',
                    help='use pre-trained model')
parser.add_argument("--dataloader_drop_last", action='store_true')
parser.add_argument('--dist-backend',
                    default='nccl',
                    type=str,
                    help='distributed backend')
parser.add_argument('--seed',
                    default=42,
                    type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
parser.add_argument('--local_rank',
                    default=-1,
                    type=int,
                    help='node rank for distributed training')
parser.add_argument(
    "--dataset_name",
    metavar="DATANAME",
    choices=["imagenet", "oxfordflowers102", "cifar10", "cifar100"],
    default="imagenet")
parser.add_argument(
    "--run_method",
    choices=["dataparallel", "wpipe"],
    default="dataparallel")
parser.add_argument("--output_dir",
                    default="output",
                    type=str,
                    help="path for output")
parser.add_argument("--chunks", type=int, help="the number of macro-batches")
parser.add_argument("--warmup_ratio",
                    default=0,
                    type=float,
                    help="the ratio of warmup")
parser.add_argument("--do_recompute",
                    action='store_true',
                    help="whether to recompute")
parser.add_argument("--do_broadcast",
                    action="store_true",
                    help="whether to use broadcast to communicate")
parser.add_argument("--do_profile",
                    action="store_true",
                    help="whether to record profile")
parser.add_argument("--do_throughput",
                    action="store_true",
                    help="whether to compute throughput")
parser.add_argument("--do_retrain",
                    action="store_true",
                    help="whether to continue training")
parser.add_argument("--retrain_times",
                    type=int,
                    default=1,
                    help="the number of retrain")
parser.add_argument("--network_config", type=str, help="network configure")
parser.add_argument("--is_scaler_fp16_p2p",
                    action="store_true",
                    help="whether to use scaler fp16 p2p")

best_acc1 = 0

num_classes_dict = {
    "imagenet": 1000,
    "cifar10": 10,
    "cifar100": 100,
    "oxfordflowers102": 102
}
INTERVAL = 30
interval_dict = {
    "imagenet": 5,
    "cifar10": 10,
    "cifar100": 10,
    "oxfordflowers102": 30
}
THROUGHPUT = 110


def main():
    args = parser.parse_args()

    global INTERVAL
    INTERVAL = interval_dict[args.dataset_name]
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )

    args.world_size = int(os.environ.get("WORLD_SIZE") or 1)

    args.distributed = args.world_size > 1 and args.gpu is None

    gpu = args.local_rank if args.distributed else args.gpu
    ngpus_per_node = torch.cuda.device_count()

    if args.distributed:
        args.rank = int(os.environ["RANK"])
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend=args.dist_backend)
    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](
            num_classes=num_classes_dict[args.dataset_name],
            norm_layer=BatchNorm2dClone)
        state_dict = load_state_dict_from_url(model_urls[args.arch])
        own_state = model.state_dict()
        for key in ("fc.weight", "fc.bias"):
            state_dict[key] = own_state[key]
        model.load_state_dict(state_dict)
    else:
        print("=> creating model '{}'".format(args.arch))
        if args.arch in models.__dict__:
            model = models.__dict__[args.arch](
                num_classes=num_classes_dict[args.dataset_name],
                norm_layer=BatchNorm2dClone)
        else:
            import re
            layers, conf, *_ = re.split("_|d", args.arch)
            num_layers = (int(layers[7:]) - 2) // 3
            base = [3, 4, 6, 3]
            times = num_layers // sum(base)
            target_layers = [i * times for i in base]
            groups, width_per_group = map(int, conf.split("x"))
            model = ResNet(Bottleneck,
                           layers=target_layers,
                           num_classes=num_classes_dict[args.dataset_name],
                           norm_layer=BatchNorm2dClone,
                           groups=groups,
                           width_per_group=width_per_group)

    if args.run_method in [
            "dataparallel",
    ]:
        main_worker(model, gpu, ngpus_per_node, args)
    else:
        main_worker_wpipe(model, ngpus_per_node, args)


def main_worker_wpipe(model, ngpus_per_node, args):
    if args.network_config is None:
        network_config = [("stage0", {
            "model": [0, 2],
            "parallel": [0, 2, 4, 6]
        }), ("stage1", {
            "model": [1, 3],
            "parallel": [1, 3, 5, 7]
        })]
    else:
        with open(args.network_config, "r") as f:
            network_config = json.load(f)["network_config"]
    get_dataset = get_dataloader_func(args.dataset_name)
    train_dataset, eval_dataset = get_dataset(args.train_batch_size,
                                              args.dataloader_num_workers,
                                              args.data_dir,
                                              args.data_dir,
                                              no_dataloader=True)
    profile_dir_name = None
    if args.do_profile:
        profile_dir_name = "profile/wpipe"
    world_size = dist.get_world_size()
    args.lr *= args.train_batch_size * world_size / 256
    dataset_step = len(train_dataset) // world_size // args.train_batch_size
    max_step = (args.num_train_epochs - args.start_epoch) * dataset_step

    def get_optimizer_wrapper(lr, momentum, weight_decay):
        def get_optimizer(model):
            optimizer = torch.optim.SGD(model.parameters(),
                                        lr,
                                        momentum=momentum,
                                        weight_decay=weight_decay)
            return optimizer

        return get_optimizer

    def get_lr_scheduler_wrapper(do_retrain, warmup_ratio, dataset_step,
                                 max_step):
        def get_lr_scheduler(optimizer):
            if do_retrain:
                return get_fixed_schedule_without_warmup(optimizer)
            lr_scheduler = get_step_schedule_with_warmup(
                optimizer,
                num_warmup_steps=int(warmup_ratio * max_step),
                num_steps_per_epoch=dataset_step)
            return lr_scheduler

        return get_lr_scheduler

    def collection_data(results):
        losses = torch.stack([item[0] for item in results])
        outputs = torch.cat([item[1] for item in results])
        labels = torch.cat([item[2] for item in results])
        return losses, outputs, labels

    def wrapper_accuracy(eval_prediction):
        preds = eval_prediction.predictions[0] if isinstance(
            eval_prediction.predictions,
            tuple) else eval_prediction.predictions
        res = accuracy(preds, eval_prediction.label_ids, topk=(1, 5))
        metrics = {"ACC1": res[0].item(), "ACC5": res[1].item()}
        return metrics

    trainer = WPipeTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=resnext_collate,
        optimizer_callbacks=(get_optimizer_wrapper(args.lr, args.momentum,
                                                   args.weight_decay),
                             get_lr_scheduler_wrapper(args.do_retrain,
                                                      args.warmup_ratio,
                                                      dataset_step, max_step)),
        callbacks=[collection_data],
        compute_metrics=wrapper_accuracy,
        based_split_method=ResnextSplit,
        backend="gloo",
        network_config=network_config,
        do_recompute=args.do_recompute,
        do_broadcast=args.do_broadcast,
        do_throughput=args.do_throughput,
        profile_dir_name=profile_dir_name,
        is_scaler_fp16_p2p=args.is_scaler_fp16_p2p)

    trainer.train()
    # eval result for each epoch
    eval_result_list = []
    if args.do_retrain:
        for i in range(args.retrain_times - 1):
            eval_result = trainer.evaluate(eval_dataset=eval_dataset)
            trainer.train(retrain=True)
            if trainer.is_world_master():
                eval_result_list.append(eval_result)

    # eval

    if not args.do_throughput:
        eval_result = trainer.evaluate(eval_dataset=eval_dataset)

        output_eval_file = os.path.join(
            args.output_dir, f"eval_results_{args.dataset_name}.txt")
        if trainer.is_world_master():
            if not os.path.exists(args.output_dir):
                os.makedirs(args.output_dir)
            with open(output_eval_file, "w") as writer:
                logger.setLevel(logging.INFO)
                logger.info("***** Eval results {} *****".format(
                    args.dataset_name))
                for key, value in eval_result.items():
                    logger.info("%s = %s", key, value)
                    writer.write("%s = %s\n" % (key, value))
            eval_result_list.append(eval_result)
            output_eval_file = os.path.join(
                args.output_dir, f"eval_all_results_{args.dataset_name}.json")
            with open(output_eval_file, "w") as fp:
                json.dump({"eval_results": eval_result_list}, fp)


def main_worker(model, gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    world_size = dist.get_world_size() if dist.is_initialized(
    ) else ngpus_per_node
    args.lr *= args.train_batch_size * world_size / 256
    optimizer = torch.optim.SGD(model.parameters(),
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(
                args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    get_dataloader = get_dataloader_func(args.dataset_name)
    train_loader, val_loader, train_sampler = get_dataloader(
        args.train_batch_size,
        args.dataloader_num_workers,
        args.data_dir,
        valdir=args.data_dir)
    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    max_step = (args.num_train_epochs - args.start_epoch) * len(train_loader)
    print_loss_obj = PrintLoss(max_step)
    data_parallel_num = world_size
    throughput_calculate_obj = ThroughPutCalculate(100, data_parallel_num,
                                                   args.train_batch_size)
    calculate_memory_obj = CalculateMemory()
    if args.do_retrain:
        lr_scheduler = get_fixed_schedule_without_warmup(optimizer)
    else:
        lr_scheduler = get_step_schedule_with_warmup(
            optimizer, int(args.warmup_ratio * max_step), len(train_loader))
    best_dict = {}
    step_count = 0
    result_list = []
    for epoch in range(args.start_epoch, args.num_train_epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        # adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        step_count = train(train_loader, model, criterion, optimizer, epoch,
                           lr_scheduler, print_loss_obj,
                           throughput_calculate_obj, step_count, args,
                           calculate_memory_obj)

        # evaluate on validation set
        acc1, acc5 = validate(val_loader, model, criterion, args)
        if args.do_throughput and step_count > THROUGHPUT:
            break
        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        result_list.append({
            "eval_ACC1": acc1.item(),
            "eval_ACC5": acc5.item()
        })
        if args.distributed and (
            (dist.is_initialized() and dist.get_rank() == 0)
                or not dist.is_initialized()):
            if not os.path.exists(args.output_dir):
                os.makedirs(args.output_dir)
            save_checkpoint(
                {
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'best_acc1': best_acc1,
                    'optimizer': optimizer.state_dict(),
                }, is_best, args.output_dir)
            if is_best:
                best_dict["best_acc1"] = best_acc1.item(),
                best_dict["best_acc5"] = acc5.item()
    best_dict["last_acc1"] = acc1.item()
    best_dict["last_acc5"] = acc5.item()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    with open(os.path.join(args.output_dir, "eval-result.json"), "w") as fp:
        json.dump(best_dict, fp)
    with open(os.path.join(args.output_dir, "eval_all_result.json"),
              "w") as fp:
        json.dump({"eval_results": result_list}, fp)
    print_loss_obj.save_loss(path=os.path.join(args.output_dir, "losses.csv"))
    throughput_calculate_obj.save_data(
        os.path.join(args.output_dir, "throughput.txt"))
    calculate_memory_obj.save_data(
        os.path.join(args.output_dir, "throughput.txt"))


def train(train_loader, model, criterion, optimizer, epoch, lr_scheduler,
          print_loss_obj, throughput_calculate_obj, step_count, args,
          calculate_memory_obj):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(train_loader),
                             [batch_time, data_time, losses, top1, top5],
                             prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, batch in enumerate(
            grouped(train_loader, args.chunks, args.run_method == "gpipe")):
        # measure data loading time
        data_time.update(time.time() - end)
        step_count += 1
        if args.run_method == "dataparallel":
            images, target = batch
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(args.gpu, non_blocking=True)
            # compute output
            output = model(images)
            loss = criterion(output, target)

        calculate_memory_obj.calculate_memory()
        throughput_calculate_obj.update(1)
        print_loss_obj.print_loss(loss, lr_scheduler.get_last_lr())
        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), target.size(0))
        top1.update(acc1[0], target.size(0))
        top5.update(acc5[0], target.size(0))
        if args.do_throughput and step_count > THROUGHPUT:
            return step_count
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        cur_rank = 0
        if dist.is_available() and dist.is_initialized():
            cur_rank = dist.get_rank()

        if i % args.print_freq == 0 and cur_rank == 0:
            progress.display(i)

    return step_count


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(val_loader), [batch_time, losses, top1, top5],
                             prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, batch in enumerate(val_loader):

            if args.run_method == "dataparallel":
                images, target = batch
                if args.gpu is not None:
                    images = images.cuda(args.gpu, non_blocking=True)
                if torch.cuda.is_available():
                    target = target.cuda(args.gpu, non_blocking=True)
                # compute output
                output = model(images)
                loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), target.size(0))
            top1.update(acc1[0], target.size(0))
            top5.update(acc5[0], target.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0 and dist.is_initialized(
            ) and dist.get_rank() == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1,
                                                                    top5=top5))

    return top1.avg, top5.avg


def save_checkpoint(state, is_best, output_dir):
    checkpoint_path = os.path.join(output_dir, 'checkpoint.pth.tar')
    torch.save(state, checkpoint_path)
    if is_best:
        shutil.copyfile(checkpoint_path,
                        os.path.join(output_dir, 'model_best.pth.tar'))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1**(epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1, 5)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        if not isinstance(output, torch.Tensor):
            output = torch.Tensor(output)
        if not isinstance(target, torch.Tensor):
            target = torch.Tensor(target)
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def reduce_mean(val):
    assert isinstance(val, torch.Tensor), "val type error!"
    if not val.is_cuda:
        val = val.cuda()
    dist.all_reduce(val)
    loss_mean = val / dist.get_world_size()
    return loss_mean.item()


def get_step_schedule_with_warmup(optimizer,
                                  num_warmup_steps,
                                  num_steps_per_epoch,
                                  last_epoch=-1):
    """
    Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0,
    after a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.

    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        num_steps_per_epoch (int):
            The number of steps of one epoch
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return 0.1**(current_step // num_steps_per_epoch // INTERVAL)

    return LambdaLR(optimizer, lr_lambda, last_epoch)


class PrintLoss(object):
    def __init__(self, par_len):
        self._pbar = None
        self.par_len = par_len
        self._loss_manager = LossManager(par_len // 120)

    def print_loss(self, outputs, lr: tuple):
        cur_rank = 0
        if self._pbar is None:
            self._pbar = DistributedProgressBar(total=self.par_len, rank=0)
        self._pbar.update(1)
        loss = outputs.detach().clone()

        if torch.distributed.is_available(
        ) and torch.distributed.is_initialized():
            cur_rank = torch.distributed.get_rank()
            torch.distributed.all_reduce(loss)
            world_size = torch.distributed.get_world_size()
        else:
            world_size = torch.cuda.device_count()
        if cur_rank == 0:
            self._loss_manager.add(loss / world_size, *lr)

    def save_loss(self, path):
        self._loss_manager.save_data(path)


if __name__ == '__main__':
    main()
