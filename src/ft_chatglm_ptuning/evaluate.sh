export NCCL_P2P_DISABLE=1
echo $NCCL_P2P_DISABLE
PRE_SEQ_LEN=128
CHECKPOINT="./experiments/outputs/PromptCBLUE-chatglm-6b-pt-128-2e-2"   # 填入用来存储模型的文件夹路径
STEP=10000    # 用来评估的模型checkpoint是训练了多少步

your_data_path="./datasets/PromptCBLUE/srchorg/"  # 填入数据集所在的文件夹路径
model_name_or_path="./models--THUDM--chatglm-6b/snapshots/a8ede826cf1b62bd3c78bdfb3625c7c5d2048fbd"   # LLM底座模型路径，或者是huggingface hub上的模型名称


CUDA_VISIBLE_DEVICES=1,2,3,4,5,6
    torchrun --nnodes 1 --nproc_per_node 6 --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:12490  \
    src/ft_chatglm_ptuning/main.py \
    --do_predict \
    --validation_file $your_data_path/dev.json \
    --test_file $your_data_path/testNullLabel.json \
    --overwrite_cache \
    --prompt_column query \
    --reference_column refs \
    --response_column answer \
    --model_name_or_path $model_name_or_path \
    --ptuning_checkpoint $CHECKPOINT/checkpoint-$STEP \
    --output_dir $CHECKPOINT \
    --overwrite_output_dir \
    --max_source_length 1500 \
    --max_target_length 256 \
    --per_device_eval_batch_size 1 \
    --predict_with_generate \
    --pre_seq_len $PRE_SEQ_LEN
