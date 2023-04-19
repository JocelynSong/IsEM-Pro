#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=32GB
#SBATCH --error=err.generate.amie.txt
#SBATCH --output=generate.amie.txt
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --time=1-0:0:0
#SBATCH --account=zhenqiaosong
#SBATCH --mail-type=fail
#SBATCH --mail-user=zhenqiao@ucsb.edu

export CUDA_VISIBLE_DEVICES=7

data_path=/mnt/data2/zhenqiaosong/protein_design/datasets/kl_test

local_root=/mnt/data/zhenqiaosong/protein_design/baselines/vae/mrf
output_path=${local_root}/amie_vae_mrf_lasso_better
generation_path=${data_path}/vae_mrf/AMIE
mkdir -p ${generation_path}

python3 fairseq_cli/generate_vae.py ${data_path} \
--arch transformer_vae_esm \
--task vae_protein_design \
--protein-task "AMIE" \
--dataset-impl "raw" \
--path ${output_path}/checkpoint_best.pt \
--batch-size 64 \
--results-path ${generation_path} \
--gen-subset test \
--sampling \
--beam 5 \
--nbest 5 \
--sampling-topk 5 \
--skip-invalid-size-inputs-valid-test
