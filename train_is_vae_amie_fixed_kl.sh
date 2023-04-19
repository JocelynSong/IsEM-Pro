#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --error=err.amie_is_vae.txt
#SBATCH --output=amie_is_vae.txt
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --time=1-0:0:0
#SBATCH --account=zhenqiaosong
#SBATCH --mail-type=fail
#SBATCH --mail-user=zhenqiao@ucsb.edu

export CUDA_VISIBLE_DEVICES=5

data_path=/mnt/data2/zhenqiaosong/protein_design/datasets/AMIE_kl
single_energy_path=${data_path}/single.lasso.pt
pair_energy_path=${data_path}/pair.lasso.pt

local_root=/mnt/data2/zhenqiaosong/protein_design/baselines/vae/vae_mrf_is
first_stage_path=/mnt/data/zhenqiaosong/protein_design/baselines/vae/mrf
pretrained_model="esm2_t6_8M_UR50D"
output_path=${local_root}/amie_vae_mrf_lasso_is_kl
first_stage_model=${first_stage_path}/amie_vae_mrf_lasso_better/checkpoint_best.pt
rm -rf ${output_path}
mkdir -p ${output_path}


python3 fairseq_cli/train_is_vae_fixed_kl.py ${data_path} \
--save-dir ${output_path} \
--restore-file ${first_stage_model} \
--task is_vae_protein_design \
--protein-task "AMIE" \
--dataset-impl "raw" \
--max-iteration-sample 600 \
--criterion importance_sampling_criterion --label-smoothing 0.1 --kl-factor 0.5 --final-kl 0.5 \
--reset-dataloader \
--arch transformer_vae_is_esm \
--single-energy-file ${single_energy_path} \
--pair-energy-file ${pair_energy_path} \
--max-length 341 \
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
--keep-interval-updates	10 \
--skip-invalid-size-inputs-valid-test \
--sampling \
--beam 5 \
--nbest 5 \
--sampling-topk 5
