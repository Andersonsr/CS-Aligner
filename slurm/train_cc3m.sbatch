#!/bin/bash
# CS-Aligner on CC3M: train_laion.py on the CC3M WebDataset shards (vlm-tuning/dataset/download_cc3m.py), validated
# on MSCOCO val2017. Submit from the CS-Aligner root:
#   sbatch slurm/train_cc3m.sbatch
#   EXPNAME=cc3m-aux0.001 sbatch slurm/train_cc3m.sbatch --aux_weight 0.001
# Paths default to the datasets home, $CLUSTER_HOME/data/dataset:
#   CC3M_ROOT  cc3m/      all of its cc3m-train-NNNN.tar shards are used
#   COCO_ROOT  coco/      val2017/ and annotations/captions_val2017.json
#   FID_STAT   coco/fid_stats_mscoco256_val2017.npz, built with make_fid_stats.py first if missing (not with
#              --no_decoder, which skips loading the Kandinsky decoder and so the sample grid and FID)
# CONDA_ENV (default dalle2) is the env activated; CLUSTER_HOME and PROXY_HOST must be set in it (conda env config
# vars). EXPNAME (default cc3m-<job id>) names the run under save/; extra arguments go to train_laion.py, after the
# defaults below so they override them. To resume, pass --resume save/<EXPNAME>/<checkpoint>.ckpt.
# Lightning reads the ranks from slurm: one task per GPU, so --ntasks-per-node must equal the gpus of --gres.
#SBATCH --job-name=cs-aligner-cc3m
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4          # one task per GPU (must equal --gres=gpu:N)
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8           # dataloader workers
#SBATCH --mem=128G
#SBATCH --time=25:00:00
#SBATCH --account=tornado
#SBATCH --output=logs/%x-%j.out
#SBATCH --partition=tc
 
 
# ESTIMATED CALIBRATED TEMPERATURES (T_2):
#   1. Softmax Formulation (Eq. 8):         T_2 = 44.4290
#   2. Normalized Log-Softmax (eta2=eta1):  T_2 = 36.6026
#   3. Absolute Log-Softmax (Loss match):   T_2 = 17.8430
#   4. Closed-Form Quadratic Approximation: T_2 = 18.7018
 
 
 
set -eo pipefail
. ~/.bashrc
conda activate "${CONDA_ENV:-dalle2}"
# proxy host and NO_PROXY are set in the conda env (conda env config vars); USER and PASSWORD come from the submit shell
export HTTP_PROXY=http://$USER:$PASSWORD@${PROXY_HOST:?set PROXY_HOST in the conda env}
export HTTPS_PROXY=$HTTP_PROXY
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
set -u
 
DATA=${CLUSTER_HOME:?set CLUSTER_HOME in the conda env}/data/dataset
CC3M_ROOT=${CC3M_ROOT:-$DATA/cc3m}
COCO_ROOT=${COCO_ROOT:-$DATA/coco}
FID_STAT=${FID_STAT:-$COCO_ROOT/fid_stats_mscoco256_val2017.npz}
EXPNAME=${EXPNAME:-cc3m-$SLURM_JOB_ID}
GPUS=${SLURM_NTASKS_PER_NODE:-1}
CPUS=${SLURM_CPUS_PER_TASK:-16}
# CC3M has ~2.9M training samples: one epoch is that many samples over all GPUs (the shards are resampled and
# --samples_per_epoch counts per process)
SAMPLES_PER_EPOCH=$((2900000 / GPUS))
 
# brace pattern over the shards (webdataset expands it): cc3m-train-{0000..0575}.tar for the 576 hf shards
NUM_SHARDS=$(find "$CC3M_ROOT" -maxdepth 1 -name 'cc3m-train-*.tar' | wc -l)
if [ "$NUM_SHARDS" -eq 0 ]; then
    echo "no cc3m-train-*.tar shards in $CC3M_ROOT"
    exit 1
fi
SHARDS="$CC3M_ROOT/cc3m-train-{0000..$(printf %04d $((NUM_SHARDS - 1)))}.tar"
 
mkdir -p logs
echo "$(date) job $SLURM_JOB_ID exp=$EXPNAME shards=$SHARDS coco=$COCO_ROOT fid=$FID_STAT gpus=$GPUS" \
    "samples/epoch/gpu=$SAMPLES_PER_EPOCH"
 
# the stats are only read by FID, which needs the decoder
if [[ " $* " != *" --no_decoder "* ]] && [ ! -f "$FID_STAT" ]; then
    echo "$(date) $FID_STAT missing, building it"
    python make_fid_stats.py --coco_root "$COCO_ROOT" --split val2017 --out "$FID_STAT" --num_workers "$CPUS"
fi
 
 
srun python train_laion.py --devices "$GPUS" --num_workers "$CPUS" --expname "$EXPNAME" \
    --laion_shards "$SHARDS" --samples_per_epoch "$SAMPLES_PER_EPOCH" \
    --coco_root "$COCO_ROOT" --val_split val2017 --fid_stat "$FID_STAT" \
    --batch_size 256 --lr 0.00005 --aux_weight 0.01 --wandb_project cs-aligner-cc3m \
    --logit_scale_init 100 --save_dir /nethome/recpinfo/users/fibz/data/checkpoint/cs-aligner/ \
    --logit_scale_schedule --logit_scale_final 44.429 --logit_scale_schedule_frac 0.70 --epochs 19 --val_every 1 "$@"
 