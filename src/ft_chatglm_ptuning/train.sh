export NCCL_P2P_DISABLE=1
echo $NCCL_P2P_DISABLE
PRE_SEQ_LEN=128
LR=2e-2
your_data_path="./datasets/PromptCBLUE/srchorg/"  # 填入数据集所在的文件夹路径
your_checkpopint_path="./experiments/outputs/"  # 填入用来存储模型的路径
model_name_or_path="./models--THUDM--chatglm-6b/snapshots/a8ede826cf1b62bd3c78bdfb3625c7c5d2048fbd"   # LLM底座模型路径，或者是huggingface hub上的模型名称

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 
    torchrun --nnodes 1 --nproc_per_node 6 --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:12490  \
    src/ft_chatglm_ptuning/main.py \
    --do_train \
    --train_file $your_data_path/train.json \
    --validation_file $your_data_path/dev.json \
    --prompt_column query \
    --reference_column refs \
    --response_column answer \
    --overwrite_cache \
    --model_name_or_path $model_name_or_path \
    --output_dir $your_checkpopint_path/PromptCBLUE-chatglm-6b-pt-$PRE_SEQ_LEN-$LR \
    --overwrite_output_dir \
    --max_source_length 1500 \
    --max_target_length 256 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 2 \
    --max_steps 10000 \
    --logging_steps 10 \
    --save_steps 100 \
    --learning_rate $LR \
    --pre_seq_len $PRE_SEQ_LEN
