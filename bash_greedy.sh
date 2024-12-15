#!/bin/bash
#
#SBATCH -A Berzelius-2024-336 -p berzelius
#SBATCH -J ADAMHENR_APR_BENCH
#SBATCH -t 08:00:00
#SBATCH -N 1 --gpus-per-node=1

# Env
container_name="elleelleaime.sif"

dataset="HumanEvalJava"
dataset_path="humaneval-java"
# dataset="Defects4J"
# dataset_path="defects4j"
# dataset="GitBugJava"
# dataset_path="gitbug-java"
method="sigonly-instruct"
sample_model_name="<<sample_model_name>>"
# patch_strategy="openai-chatcompletion"
# candidate_model_name="gpt-4o-mini"
# patch_strategy="codellama-instruct"
# candidate_model_name="meta-llama/CodeLlama-7b-Instruct-hf"
patch_strategy="google"
candidate_model_name="gemini-1.5-flash"


samples_path="../APR-benchmark-results/data/${dataset_path}/samples_${dataset}_${method}_.jsonl"
kwargs="--model_name $candidate_model_name --max_length=4096 --generation_strategy beam_search --num_return_sequences 1"
script="python generate_patches.py $samples_path $patch_strategy $kwargs"
apptainer exec --nv $container_name $script
# echo $script