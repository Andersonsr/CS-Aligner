#!/bin/bash
# Image-text retrieval (eval_retrieval.py) on MSCOCO val2017, with the trained prior(s) and with plain CLIP ViT-bigG
# for reference. Submit from the CS-Aligner root:
#   sbatch slurm/eval_retrieval.sbatch --ckpt save/<exp>/<checkpoint>.ckpt
#   sbatch slurm/eval_retrieval.sbatch --ckpt save/<exp>/*.ckpt save/<exp2>/*.ckpt   # several trained models
#   sbatch slurm/eval_retrieval.sbatch                   # CLIP ViT-bigG only
# COCO_ROOT (default $CLUSTER_HOME/data/dataset/coco) holds val2017/ and annotations/captions_val2017.json.
# CONDA_ENV (default dalle2) is the env activated; CLUSTER_HOME and PROXY_HOST must be set in it (conda env config
# vars). Extra arguments go to eval_retrieval.py; the results go to <checkpoint without .ckpt>_retrieval.json with one
# checkpoint, retrieval.json next to the first checkpoint with several, retrieval_clip.json without any, unless --out
# is given.
#SBATCH --job-name=cs-aligner-retrieval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16           # dataloader workers
#SBATCH --mem=64G
#SBATCH --time=02:00:00
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

COCO_ROOT=${COCO_ROOT:-${CLUSTER_HOME:?set CLUSTER_HOME in the conda env}/data/dataset/coco}

mkdir -p logs
echo "$(date) job $SLURM_JOB_ID coco=$COCO_ROOT args=$* gpus=${CUDA_VISIBLE_DEVICES:-none}"

srun python eval_retrieval.py --coco_root "$COCO_ROOT" --num_workers "${SLURM_CPUS_PER_TASK:-16}" "$@"
