#!/bin/bash
# Zero-shot classification (eval_zeroshot.py) on the test sets of clip-refine, with the trained prior and with plain
# CLIP ViT-bigG for reference. Submit from the CS-Aligner root:
#   sbatch slurm/eval_zeroshot.sbatch --ckpt save/<exp>/<checkpoint>.ckpt
#   sbatch slurm/eval_zeroshot.sbatch --ckpt save/<exp>/*.ckpt save/<exp2>/*.ckpt   # several trained models
#   sbatch slurm/eval_zeroshot.sbatch                   # CLIP ViT-bigG only
# With several checkpoints the images are encoded once, each checkpoint is a run "<exp>/<checkpoint name>", and all
# of them go to one zeroshot.json next to the first checkpoint (or --out).
#   sbatch slurm/eval_zeroshot.sbatch --ckpt ... --datasets zeroshot_classification_imagenet
# DATA_ROOT (default $CLUSTER_HOME/data/dataset, the datasets home) is the folder given to vlm-tuning's
# download_datasets.sh --data. CONDA_ENV (default dalle2) is the env activated; CLUSTER_HOME and PROXY_HOST must be set
# in it (conda env config vars). Extra arguments go to eval_zeroshot.py; the results go to
# <checkpoint without .ckpt>_zeroshot.json (zeroshot_clip.json without --ckpt) unless --out is given.
#SBATCH --job-name=cs-aligner-zeroshot
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16           # dataloader workers
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --account=tornado
#SBATCH --output=logs/cs-aligner-%x-%j.out
##SBATCH --partition=tc

set -eo pipefail
. ~/.bashrc
conda activate "${CONDA_ENV:-dalle2}"
# proxy host and NO_PROXY are set in the conda env (conda env config vars); USER and PASSWORD come from the submit shell
export HTTP_PROXY=http://$USER:$PASSWORD@${PROXY_HOST:?set PROXY_HOST in the conda env}
export HTTPS_PROXY=$HTTP_PROXY
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
set -u

DATA_ROOT=${DATA_ROOT:-${CLUSTER_HOME:?set CLUSTER_HOME in the conda env}/data/dataset}   # datasets home

mkdir -p logs
echo "$(date) job $SLURM_JOB_ID data=$DATA_ROOT args=$* gpus=${CUDA_VISIBLE_DEVICES:-none}"

srun python eval_zeroshot.py --data_root "$DATA_ROOT" --num_workers "${SLURM_CPUS_PER_TASK:-16}" "$@"
