# 1 machine * 8 gpus

package_root=$(dirname "$0")
export PYTHONPATH=$package_root:$PYTHONPATH

data_dir=/home/myself/wpipe/data/dataset/glue

# 96,192
num_layers=96

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
--arch=bert \
--num_layers=$num_layers \
--depth $pipeline_depth \
--num_data_parallel $data_parallel \
--method wpipe \
--batch_size $batch_size \
--do_recompute \
--task nlp
