#! /bin/bash

# Change for multinode config
MP_SIZE=2
EP_SIZE=4

NUM_WORKERS=2
NUM_GPUS_PER_WORKER=8

script_path=$(realpath $0)
script_dir=$(dirname $script_path)

config_json="$script_dir/ds_zero2_config.json"
gpt_options=" \
       --model-parallel-size ${MP_SIZE} \
       --expert-parallel-size ${EP_SIZE} \
       --num-layers 4 \
       --hidden-size 512 \
       --num-attention-heads 16 \
       --batch-size 8 \
       --seq-length 1024 \
       --max-position-embeddings 1024 \
       --train-iters 1000 \
       --resume-dataloader \
       --train-data /data/megatron/webtext/data.json \
       --lazy-loader \
       --tokenizer-type GPT2BPETokenizer \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr 0.00015 \
       --no-load-optim \
       --lr-decay-style cosine \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --warmup .01 \
       --fp16 \
       --num-experts 16
       --log-interval 10
       --text-key text \
       --loose-json
"

# Disable activation checkpointing

#     --checkpoint-activations \
#       --deepspeed-activation-checkpointing \

gpt_options="${gpt_options}
               --deepspeed \
               --deepspeed_config ${config_json} \
"


run_cmd="LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libnccl.so.2.8.3 deepspeed --num_nodes ${NUM_WORKERS} --num_gpus ${NUM_GPUS_PER_WORKER} pretrain_gpt2.py $@ ${gpt_options}"
echo ${run_cmd}
eval ${run_cmd}

set +x
