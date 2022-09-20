#!/usr/bin/env bash

PYT=/home/ec2-user/anaconda3/envs/xdpr/bin/python

percent=$1
version=$2
ckpt=$3 
output_dir=retriever_v${version}_clean
MODEL="/home/ec2-user/quic-efs/user/yifangao/SearchQURetrievalKeyphraseGeneration/ckpts/percent_${percent}/${output_dir}/dpr_biencoder.${ckpt}"
$PYT generate_dense_embeddings.py \
  model_file=${MODEL} \
  ctx_src=dpr_kp \
  out_file="${MODEL}.us" \
  batch_size=5120 \
  encoder.share_encoder=True




