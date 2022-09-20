#!/usr/bin/env bash

function run_ddp() {
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export OMP_NUM_THREADS=1
N_GPU=8
OUTPUT_DIR=$1
EP=10
LR=1e-4
RE_ENKPS=$2
CKPT=$3
TEST_FILE=$4
PERCENT=$5

DATA="/home/ec2-user/quic-efs/user/yifangao/AmazonMultilingualKeyphraseDataset/percent-${PERCENT}"

TEST="${DATA}/${TEST_FILE}"

PYT=/home/ec2-user/anaconda3/envs/rakg/bin/python
$PYT -m torch.distributed.launch \
--nproc_per_node ${N_GPU} \
--master_port=4684 train_multilingual_kpgen.py \
--predict_with_generate \
--output_dir=${OUTPUT_DIR}/ddp_${N_GPU}_gpu_e${EP}_lr${LR} \
--overwrite_output_dir \
--do_predict \
--evaluation_strategy=epoch \
--per_device_eval_batch_size=12 \
--learning_rate=${LR} \
--weight_decay=0.01 \
--num_train_epochs=${EP} \
--warmup_steps=100 \
--logging_steps=200 \
--seed=95 \
--fp16 \
--load_best_model_at_end \
--metric_for_best_model=eval_f1 \
--greater_is_better=True \
--model_name_or_path=facebook/mbart-large-cc25 \
--source_lang=ar_AR \
--target_lang=ar_AR \
--test_file=${TEST} \
--overwrite_cache \
--num_beams=5 \
--retrieval_augmented_generation \
--retrieved_kp_file=${RE_ENKPS} \
--model_path_for_inference="${OUTPUT_DIR}/ddp_${N_GPU}_gpu_e${EP}_lr${LR}/checkpoint-${CKPT}"
}


TOPK=5
PERCENT=$1
TEST_FILE=mix.train.nonaligned.json

VERSION=$2
CKPT=$3

HOME="/home/ec2-user/quic-efs/user/yifangao/SearchQURetrievalKeyphraseGeneration/ckpts/percent_${PERCENT}/generator_v${VERSION}"
OUTPUT_DIR="${HOME}/rakg_top${TOPK}_aligned"
RE_ENKPS="${HOME}/asin2keyphrase.top${TOPK}.json"
run_ddp ${OUTPUT_DIR} ${RE_ENKPS} ${CKPT} ${TEST_FILE} ${PERCENT}


