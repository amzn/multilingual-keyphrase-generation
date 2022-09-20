#!/usr/bin/env bash

PYT=/home/ec2-user/anaconda3/envs/xdpr/bin/python
export HYDRA_FULL_ERROR=1
export OC_CAUSE=1
export OMP_NUM_THREAD=1

percent=$1
version=$2
pretrain_ckpt=$3
output_dir=retriever_v${version}_clean
OUTPUT="/home/ec2-user/quic-efs/user/yifangao/SearchQURetrievalKeyphraseGeneration/ckpts/percent_${percent}/${output_dir}"
pretrain_output_dir=retriever_v${version}_noisy
PT_MODEL="/home/ec2-user/quic-efs/user/yifangao/SearchQURetrievalKeyphraseGeneration/ckpts/percent_${percent}/${pretrain_output_dir}/dpr_biencoder.${pretrain_ckpt}"
$PYT -m torch.distributed.launch --nproc_per_node=8 train_dense_encoder.py \
  train.batch_size=8 \
  output_dir=${OUTPUT} \
  train.log_batch_step=500 \
  train.train_rolling_loss_step=500 \
  encoder.share_encoder=True \
  train_datasets=[mkp_${percent}_de_train_aligned,mkp_${percent}_fr_train_aligned,mkp_${percent}_es_train_aligned,mkp_${percent}_it_train_aligned] \
  dev_datasets=[mkp_${percent}_mix_dev] \
  train.num_train_epochs=15 \
  train.val_av_rank_hard_neg=15 \
  model_file=${PT_MODEL} \
  ignore_checkpoint_offset=True \
  ignore_checkpoint_optimizer=True



