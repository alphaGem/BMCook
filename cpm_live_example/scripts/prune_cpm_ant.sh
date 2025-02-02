#! /bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
GPUS_PER_NODE=4

NNODES=1
MASTER_ADDR=localhost
MASTER_PORT=2333

OPTS=""
OPTS+=" --model-config cpm_live_example/config/cpm-ant-1b.json"
OPTS+=" --batch-size 32"
OPTS+=" --train-iters 200000"
OPTS+=" --save-iters 500"
OPTS+=" --save-name cpm_live_checkpoint"
OPTS+=" --max-length 512"
OPTS+=" --save results/"
OPTS+=" --lr 0.1"
OPTS+=" --inspect-iters 100"
OPTS+=" --warmup-iters 2000"
OPTS+=" --lr-decay-style noam"
OPTS+=" --weight-decay 0.01"
OPTS+=" --clip-grad 1.0"
OPTS+=" --loss-scale 1048576"
OPTS+=" --start-step 0"
OPTS+=" --log-dir logs/tensorboard/cpm_live_48_4096/"
OPTS+=" --load cpm_live_compress/ckpt/1B/cpm_live_checkpoint_1B_pruned.pt"
OPTS+=" --data_bin_path /path/to/your/dataset"

CMD="torchrun --nnodes=${NNODES} --nproc_per_node=${GPUS_PER_NODE} --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} prune_cpm_ant.py ${OPTS}"

echo ${CMD}
$CMD

