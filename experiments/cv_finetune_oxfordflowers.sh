# Fine-tune

package_root=$(dirname "$0")
export PYTHONPATH=$package_root:$PYTHONPATH

data_dir=/home/myself/wpipe/data/oxford_flowers

# resnext50_32x8d, resnext101_32x8d
model=resnext50_32x8d

# oxfordflowers102
dataset=oxfordflowers102

epochs=95

python -m torch.distributed.launch --nproc_per_node 8 cv/resnext_main.py \
--data_dir $data_dir \
--arch $model \
--dist-backend 'nccl' \
-j 20 \
-b 32 \
--lr 0.01 \
--dataset_name $dataset \
--num_train_epochs $epochs \
--output_dir outputs \
--seed 42 \
--run_method wpipe
