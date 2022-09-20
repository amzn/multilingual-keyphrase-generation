#!/usr/bin/env bash

export HYDRA_FULL_ERROR=1
export OC_CAUSE=1
export OMP_NUM_THREAD=1

PYT=/home/ec2-user/anaconda3/envs/xdpr/bin/python

percent=$1

previous_version=$2
version=$3
output_dir=retriever_v${version}_noisy
OUTPUT="/home/ec2-user/quic-efs/user/yifangao/SearchQURetrievalKeyphraseGeneration/ckpts/percent_${percent}/${output_dir}"
$PYT -m torch.distributed.launch --nproc_per_node=8 train_dense_encoder.py \
  train.batch_size=8 \
  output_dir=${OUTPUT} \
  train.log_batch_step=500 \
  train.train_rolling_loss_step=500 \
  encoder.share_encoder=True \
  train_datasets=[mkp_${percent}_mix_train_nonaligned_pseudo_label_v${previous_version}] \
  dev_datasets=[mkp_${percent}_mix_dev] \
  train.num_train_epochs=15 \
  train.val_av_rank_hard_neg=15


