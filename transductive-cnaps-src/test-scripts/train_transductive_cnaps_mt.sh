#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH --partition=edith
#SBATCH --cpus-per-task=5
#SBATCH --mem=30G
#SBATCH --nodes=1
#SBATCH --time=3-00:00

REPO_ROOT="/ubc/cs/research/plai-scratch/peyman/new_cnaps_results/simple-cnaps/transductive-cnaps-src/"
cd "${REPO_ROOT}"


conda deactivate
deactivate
source /ubc/cs/research/plai-scratch/peyman/new_cnaps_results/simple-cnaps-tests/simple-meta-dataset/new-env/bin/activate

ulimit -n
ulimit -n 5000
ulimit -n

export META_DATASET_ROOT=/ubc/cs/research/plai-scratch/peyman/meta-dataset-repo/
export PYTHONPATH=/ubc/cs/research/plai-scratch/peyman/meta-dataset-repo/

python -u run_transductive_cnaps_mt.py \
        --feature_adaptation $1 \
        --checkpoint_dir ../checkpoints/$2_shots_$3_ways_on_$4_dataset/ \
        --shots $2 \
	--ways $3 \
        --dataset $4
