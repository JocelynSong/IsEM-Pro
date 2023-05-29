#!/bin/bash

data_path=$1
protein=$2
single_energy_path=${protein}/single.lasso.pt
pair_energy_path=${protein}/pair.lasso.pt

output_path=$3
first_stage_path=$4
pretrained_model="esm2_t6_8M_UR50D"
first_stage_model=${first_stage_path}/checkpoint_best.pt
rm -rf ${output_path}
mkdir -p ${output_path}


python3 fairseq_cli/train_is_vae.py ${data_path} \
--save-dir ${output_path} \
--restore-file ${first_stage_model} \
--task is_vae_protein_design \
--protein-task ${protein} \
--dataset-impl "raw" \
--max-iteration-sample 3600 \
--criterion importance_sampling_criterion --label-smoothing 0.1 --kl-factor 0.8 --final-kl 0.8 \
--reset-dataloader \
--arch transformer_vae_is_esm \
--single-energy-file ${single_energy_path} \
--pair-energy-file ${pair_energy_path} \
--max-length 75 \
--encoder-embed-dim 320 \
--decoder-embed-dim 320 \
--decoder-layers 2 \
--pretrained-esm-model ${pretrained_model} \
--latent-dimension 320 \
--latent-sample-size 1 \
--dropout 0.3 \
--share-decoder-input-output-embed \
--optimizer adam --adam-betas '(0.9,0.98)' \
--lr 1e-5 --lr-scheduler inverse_sqrt \
--stop-min-lr '1e-08' --warmup-updates 4000 \
--warmup-init-lr '1e-06' \
--weight-decay 0.01 \
--clip-norm 0.01 \
--ddp-backend legacy_ddp \
--log-format 'simple' --log-interval 10 \
--max-tokens 4096 \
--update-freq 1 \
--max-update 300000 \
--max-epoch 10 \
--fp16 \
--valid-subset valid \
--max-sentences-valid 8 \
--validate-interval 1 \
--save-interval 1 \
--validate-after-updates 3000 \
--validate-interval-updates 3000 \
--save-interval-updates 3000 \
--keep-interval-updates 10 \
--skip-invalid-size-inputs-valid-test \
--sampling \
--beam 5 \
--nbest 5 \
--sampling-topk 5
