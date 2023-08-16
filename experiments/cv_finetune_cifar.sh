# Fine-tune
package_root=$(dirname "$0")
export PYTHONPATH=$package_root:$PYTHONPATH

data_dir=/output/public_datasets/cv/cifar

# 	Experiments
# -------------------------------
# model			dataset		epochs
# resnext50_32x8d	cifar100	35
# resnext50_32x8d	cifar10		35
# resnext101_32x8d	cifar100	35
# resnext101_32x8d	cifar10		35

# change this parameters according to the table above
model=resnext50_32x8d
dataset=cifar100
epochs=35

python -m torch.distributed.launch --nproc_per_node 8 cv/resnext_main.py \
--data_dir $data_dir \
--arch resnext50_32x4d \
--dist-backend 'nccl' \
-j 20 \
-b 32 \
--lr 0.01 \
--dataset_name $dataset \
--num_train_epochs $epochs \
--output_dir outputs \
--seed 42 \
--run_method wpipe
