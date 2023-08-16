# 8 nodes * 8 gpus

package_root=$(dirname "$0")
export PYTHONPATH=$package_root:$PYTHONPATH

data_dir=/home/myself/wpipe/data/oxford_flowers

# resnext500_32x4d, resnext800_32x4d 
model=resnext500_32x4d

# (64, 1),(32,2),(16,4),(8,8),(4,16)
pipeline_depth=16
data_parallel=4

# 1,2,4,8,16,32,64,128,256,512
batch_size=32

#sh tool/init.sh
python tool/run.py  \
--node_rank $RANK \
--master_addr $MASTER_ADDR \
--nnodes 8 \
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
