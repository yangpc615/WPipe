# wpipe,cifar,scartch

package_root=$(dirname "$0")
export PYTHONPATH=$package_root:$PYTHONPATH

data_dir=/output/public_datasets/cv/cifar
model=resnext50_32x8d
dataset=cifar100
epochs=1

python -m torch.distributed.launch --nproc_per_node  8 cv/resnext_main.py  \
--data_dir $data_dir \
--arch $model \
--dist-backend nccl \
-j 20 \
-b 32 \
--lr 0.01 \
--dataset_name $dataset \
--num_train_epochs $epochs \
--output_dir outputs \
--seed 42 \
--run_method wpipe \
--do_retrain \
--retrain_times 30

