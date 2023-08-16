# 1 machine * 8 gpus

package_root=$(dirname "$0")
export PYTHONPATH=$package_root:$PYTHONPATH

data_dir=/home/myself/wpipe/data/oxford_flowers

# resnext50_32x4d, resnext101_32x4d, resnext152_32x4d
# resnext200_32x4d, resnext302_32x4d 
model=resnext200_32x4d

# (8, 1),(4,2),(2,4)
pipeline_depth=8
data_parallel=1

# 1,2,4,8,16,32,64,128,256,512
batch_size=32

#sh tool/init.sh
python tool/run.py  \
--node_rank 0 \
--master_addr localhost \
--nnodes 1 \
--nproc_per_node 8 \
--master_port 29500 \
--data_dir $data_dir \
--arch=$model \
--depth $pipeline_depth \
--num_data_parallel $data_parallel \
--method wpipe \
--batch_size $batch_size \
--do_recompute \
--task cv
