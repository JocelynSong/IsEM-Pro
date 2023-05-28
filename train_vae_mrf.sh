#!/bin/bash

data_path=$1
single_energy_path=${data_path}/Pab1/single.lasso.pt
pair_energy_path=${data_path}/Pab1/pair.lasso.pt

pretrained_model="esm2_t6_8M_UR50D"
output_path=$2


python3 fairseq_cli/train.py ${data_path} \
--save-dir ${output_path} \
--task vae_protein_design \
--protein-task "Pab1" \
--dataset-impl "raw" \
--criterion vae_transformer_criterion --label-smoothing 0.1 --kl-factor 0.8 --final-kl 0.8 \
--arch transformer_vae_esm \
--single-energy-file ${single_energy_path} \
--pair-energy-file ${pair_energy_path} \
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
--max-epoch 30 \
--fp16 \
--valid-subset valid \
--max-sentences-valid 8 \
--validate-interval 1 \
--save-interval 1 \
--validate-after-updates 3000 \
--validate-interval-updates 3000 \
--save-interval-updates 3000 \
--keep-interval-updates 10 \
--skip-invalid-size-inputs-valid-test