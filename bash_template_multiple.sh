#!/bin/bash
#
#SBATCH -A NAISS2023-5-388 -p alvis
#SBATCH -J ADAMHENR_APR_BENCH
#SBATCH -t 08:00:00
#SBATCH -N 1 --gpus-per-node=A100:4

# Env
container_name="elleelleaime.sif"
export TRANSFORMERS_CACHE="/cephyr/users/adamhenr/Alvis/mimer/elle-elle-aime/.transformers_cache"

dataset="<<dataset>>"
method="<<method>>"
sample_model_name="<<sample_model_name>>"
patch_strategy="<<patch_strategy>>"
candidate_model_name="<<candidate_model_name>>"
temp="<<temperature>>"

samples_path="samples_${dataset}_${method}_.jsonl.gz"
kwargs="--model_name $candidate_model_name --n_workers 1 --temperature $temp --generation_strategy sampling --num_return_sequences 10"
script="python generate_patches.py $samples_path $patch_strategy $kwargs"
apptainer exec --nv $container_name $script
# echo $script