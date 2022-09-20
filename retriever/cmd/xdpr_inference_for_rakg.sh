#!/usr/bin/env bash

PYT=/home/ec2-user/anaconda3/envs/xdpr/bin/python

export CUDA_VISIBLE_DEVICES=$1

percent=$1
version=$2
ckpt=$3
output_dir=retriever_v${version}_clean
model_path="/home/ec2-user/quic-efs/user/yifangao/SearchQURetrievalKeyphraseGeneration/ckpts/percent_${percent}/${output_dir}/dpr_biencoder.${ckpt}"

lang="${percent}_mix_train_dev_test_full"

echo "===$lang==="
$PYT dense_retriever.py \
qa_dataset=mkp_${lang} \
ctx_datatsets=[dpr_kp] \
encoded_ctx_files=["${model_path}.us_0"] \
out_file="${model_path}.us_0.retrieved_${lang}" \
model_file=${model_path} \
encoder.share_encoder=True


