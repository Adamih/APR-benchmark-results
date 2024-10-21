#!/bin/bash
#
#SBATCH -A NAISS2023-5-388 -p alvis
#SBATCH -J ADAMHENR_APR_BENCH
#SBATCH -t 01:00:00
#SBATCH -N 1 --gpus-per-node=A100:2

# Env
container_name="elleelleaime.sif"
export TRANSFORMERS_CACHE="/cephyr/users/adamhenr/Alvis/mimer/elle-elle-aime/.transformers_cache"

dataset="<<dataset>>"
method="<<method>>"
sample_model_name="<<sample_model_name>>"
candidate_model_name="<<candidate_model_name>>"
temp="<<temp>>"

samples_path="samples_${dataset}_${method}_model_name_${sample_model_name}.jsonl.gz"
kwargs="--n_workers 1 --temperature $temp --generation_strategy beam_search --num_beams 10 --num_return_sequences 10"
script="python generate_patches.py $samples_path $candidate_model_name $kwargs"
apptainer exec --nv $container_name $script
# echo $script