#!/bin/bash

#SBATCH --gres=gpu:2
#SBATCH --partition=plai
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --nodes=1
#SBATCH --job-name=transductive_cnaps_active_learning_cifar10_all_algos
#SBATCH --output=%x-%a.out
#SBATCH --time=3-00:00
#SBATCH --array=11

REPO_ROOT="/ubc/cs/research/plai-scratch/peyman/new_cnaps_results/simple-cnaps/active-learning/"
cd "${REPO_ROOT}"

ulimit -n
ulimit -n 5000
ulimit -n

conda deactivate
deactivate
source /ubc/cs/research/plai-scratch/peyman/new_cnaps_results/simple-cnaps-tests/simple-meta-dataset/new-env/bin/activate

export META_DATASET_ROOT=/ubc/cs/research/plai-scratch/peyman/new_cnaps_results/meta-dataset/
export PYTHONPATH=/ubc/cs/research/plai-scratch/peyman/new_cnaps_results/meta-dataset/

echo "\n\n\npredictive_entropy\n\n\n"

python3 -u run_active_learning.py \
        --data_path /ubc/cs/research/plai-scratch/peyman/new_cnaps_results/active-learning/data \
        --feature_adaptation film \
        --checkpoint_dir ../checkpoints/${SLURM_JOB_ID} \
        --model transductive_cnaps \
        --dataset cifar10 --test_model_path ../model-checkpoints/meta-dataset-checkpoints/best_transductive_cnaps.pt --active_learning_method predictive_entropy

echo "\n\n\nvar_ratios\n\n\n"

python3 -u run_active_learning.py \
        --data_path /ubc/cs/research/plai-scratch/peyman/new_cnaps_results/active-learning/data \
        --feature_adaptation film \
        --checkpoint_dir ../checkpoints/${SLURM_JOB_ID} \
        --model transductive_cnaps \
        --dataset cifar10 --test_model_path ../model-checkpoints/meta-dataset-checkpoints/best_transductive_cnaps.pt --active_learning_method var_ratios

echo "\n\n\nrandom\n\n\n"

python3 -u run_active_learning.py \
        --data_path /ubc/cs/research/plai-scratch/peyman/new_cnaps_results/active-learning/data \
        --feature_adaptation film \
        --checkpoint_dir ../checkpoints/${SLURM_JOB_ID} \
        --model transductive_cnaps \
        --dataset cifar10 --test_model_path ../model-checkpoints/meta-dataset-checkpoints/best_transductive_cnaps.pt --active_learning_method random
