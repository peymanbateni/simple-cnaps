#!/bin/sh

#SBATCH --gres=gpu:2
#SBATCH --partition=edith
#SBATCH --cpus-per-task=10
#SBATCH --mem=60G
#SBATCH --nodes=1
#SBATCH --job-name=simple_cnaps_continual_learning_first
#SBATCH --output=%x-%a.out
#SBATCH --time=3-00:00
#SBATCH --array=1

REPO_ROOT="/ubc/cs/research/plai-scratch/peyman/new_cnaps_results/simple-cnaps/continual-learning/"
cd "${REPO_ROOT}"

conda deactivate
deactivate
source /ubc/cs/research/plai-scratch/peyman/new_cnaps_results/simple-cnaps-tests/simple-meta-dataset/new-env/bin/activate

STRATEGY="first-encoding"
MODEL="simple_cnaps"

alias run_continual_mnist="python -u run_continual_learning.py --dataset MNIST --test_shot 1000 --test_epochs 30 --model $MODEL --continual_learning_strategy $STRATEGY"
alias run_continual_cifar10="python -u run_continual_learning.py --dataset CIFAR10 --test_shot 1000 --test_epochs 30 --CIFAR_resize on --model $MODEL --continual_learning_strategy $STRATEGY"
alias run_continual_cifar100="python -u run_continual_learning.py --dataset CIFAR100 --test_shot 1000 --test_epochs 30 --CIFAR_resize on --model $MODEL --continual_learning_strategy $STRATEGY"

echo "run_continual_mnist"

# MNIST experiments
run_continual_mnist --shot 1 --head_type multi
run_continual_mnist --shot 5 --head_type multi
run_continual_mnist --shot 10 --head_type multi
run_continual_mnist --shot 25 --head_type multi
run_continual_mnist --shot 100 --head_type multi
# run_continual_mnist --shot 1000 --head_type multi

run_continual_mnist --shot 1 --head_type single
run_continual_mnist --shot 5 --head_type single
run_continual_mnist --shot 10 --head_type single
run_continual_mnist --shot 25 --head_type single
run_continual_mnist --shot 100 --head_type single
# run_continual_mnist --shot 1000 --head_type single

echo "run_continual_cifar10"

# CIFAR10 experiments
run_continual_cifar10 --shot 1 --head_type multi
run_continual_cifar10 --shot 5 --head_type multi
run_continual_cifar10 --shot 10 --head_type multi
run_continual_cifar10 --shot 25 --head_type multi
run_continual_cifar10 --shot 100 --head_type multi
# run_continual_cifar10 --shot 1000 --head_type multi

run_continual_cifar10 --shot 1 --head_type single
run_continual_cifar10 --shot 5 --head_type single
run_continual_cifar10 --shot 10 --head_type single
run_continual_cifar10 --shot 25 --head_type single
run_continual_cifar10 --shot 100 --head_type single
# run_continual_cifar10 --shot 1000 --head_type single

echo "run_continual_cifar100"

# CIFAR100 experiments
run_continual_cifar100 --shot 1 --head_type multi --continual_time_steps 10
run_continual_cifar100 --shot 5 --head_type multi --continual_time_steps 10
run_continual_cifar100 --shot 10 --head_type multi --continual_time_steps 10
run_continual_cifar100 --shot 25 --head_type multi --continual_time_steps 10
run_continual_cifar100 --shot 100  --head_type multi --continual_time_steps 10
# run_continual_cifar100 --shot 1000 --head_type multi --continual_time_steps 10

run_continual_cifar100 --shot 1 --head_type single --continual_time_steps 10
run_continual_cifar100 --shot 5 --head_type single --continual_time_steps 10
run_continual_cifar100 --shot 10 --head_type single --continual_time_steps 10
run_continual_cifar100 --shot 25 --head_type single --continual_time_steps 10
run_continual_cifar100 --shot 100 --head_type single --continual_time_steps 10
# run_continual_cifar100 --shot 1000 --head_type single --continual_time_steps 10
