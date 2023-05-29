#!/bin/bash

data_path=$1
protein=$2
output_path=$3
generation_path=$4
mkdir -p ${generation_path}

python3 fairseq_cli/generate_vae.py ${data_path} \
--arch transformer_vae_esm \
--task vae_protein_design \
--protein-task ${protein} \
--dataset-impl "raw" \
--path ${output_path}/checkpoint_best.pt \
--batch-size 128 \
--results-path ${generation_path} \
--gen-subset test \
--sampling \
--beam 5 \
--nbest 5 \
--sampling-topk 5 \
--skip-invalid-size-inputs-valid-test
