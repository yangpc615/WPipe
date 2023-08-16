# scratch

package_root=$(dirname "$0")
export PYTHONPATH=$package_root:$PYTHONPATH

task_name=QQP
data_dir=/home/myself/wpipe/data/dataset/glue/$task_name

python -m torch.distributed.launch --nproc_per_node 8 nlp/bert_main.py \
--model_name_or_path bert-base-uncased \
--task_name $task_name \
--do_train \
--do_eval \
--data_dir $data_dir \
--max_seq_length 128 \
--per_device_train_batch_size 32 \
--learning_rate 5e-6 \
--num_train_epochs 1.0 \
--seed 42 \
--output_dir outputs \
--overwrite_output_dir \
--dataloader_drop_last \
--do_retrain \
--retrain_times 30
