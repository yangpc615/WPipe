# 8 nodes * 8 gpus

package_root=$(dirname "$0")
export PYTHONPATH=$package_root:$PYTHONPATH

data_dir=/home/myself/wpipe/data/dataset/glue

# 384,768
num_layers=384

# (64,1),(32,2),(16,4),(8,8)
pipeline_depth=32
data_parallel=2

# 1,2,4,8,16,32,64,128,256,512
batch_size=128

#sh tool/init.sh
python tool/run.py  \
--node_rank $RANK \
--master_addr $MASTER_ADDR \
--nnodes 8 \
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
