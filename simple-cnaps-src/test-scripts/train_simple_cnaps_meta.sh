#!/bin/bash

#SBATCH --gres=gpu:2
#SBATCH --partition=plai
#SBATCH --cpus-per-task=10
#SBATCH --mem=60G
#SBATCH --nodes=1
#SBATCH --job-name=meta_simple
#SBATCH --output=%x-%a.out
#SBATCH --time=3-00:00
#SBATCH --array=1

REPO_ROOT="/ubc/cs/research/plai-scratch/peyman/new_cnaps_results/simple-cnaps/simple-cnaps-src"
cd "${REPO_ROOT}"

conda deactivate
deactivate
source /ubc/cs/research/plai-scratch/peyman/new_cnaps_results/simple-cnaps-tests/simple-meta-dataset/new-env/bin/activate

ulimit -n
ulimit -n 5000
ulimit -n

export META_DATASET_ROOT=/ubc/cs/research/plai-scratch/peyman/new_cnaps_results/simple-cnaps/meta-dataset/
export PYTHONPATH=/ubc/cs/research/plai-scratch/peyman/new_cnaps_results/simple-cnaps/meta-dataset/

python3.7 -u run_simple_cnaps.py \
        --data_path /ubc/cs/research/plai-scratch/peyman/meta-dataset/records/ \
        --feature_adaptation film \
        --checkpoint ./checkpoints/${SLURM_JOB_NAME}/
