# Fine-tune

package_root=$(dirname "$0")
export PYTHONPATH=$package_root:$PYTHONPATH


data_dir=/home/myself/wpipe/data/dataset/glue

# 				Experiments 
# ------------------------------------------------------------------------------------
# task	model			batch_size	learning_rate	warmup_steps	epochs
# MNLI	bert-base-uncased	32		8e-5		1000		15.0
# MNLI	bert-large-uncased	16		4e-5		2000		15.0
# QQP	bert-base-uncased	32		8e-5		1000		15.0
# QQP	bert-large-uncased	16		4e-5		2000		15.0
# ------------------------------------------------------------------------------------

# change this parameters according to the table above
task_name=MNLI
model=bert-base-uncased
batch_size=32
learning_rate=8e-5
warmup_steps=1000
epochs=15.0

python -m torch.distributed.launch --nproc_per_node 8 nlp/bert_main.py \
--model_name_or_path bert-base-uncased \
--task_name $task_name \
--do_train \
--do_eval \
--data_dir $data_dir/$task_name \
--max_seq_length 128 \
--per_device_train_batch_size $batch_size \
--learning_rate $learning_rate \
--warmup_steps $warmup_steps \
--num_train_epochs $epochs \
--seed 42 \
--output_dir outputs \
--overwrite_output_dir \
--dataloader_drop_last
