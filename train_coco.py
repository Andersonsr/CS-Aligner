"""Train CS-Aligner on MSCOCO captions (train2014), validate on val2014.

Launch (see scripts/run_coco.sh):
    torchrun --nproc_per_node=2 train_coco.py --devices 2 --batch_size 1024 ...
"""
import argparse
import os

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CocoCaptions

from cs_aligner.lightning_module import CSAligner, add_common_args
from cs_aligner.model import CSAlignerBackbone


def parse_args():
    parser = argparse.ArgumentParser(description="CS-Aligner on MSCOCO")
    add_common_args(parser)
    parser.add_argument("--coco_root", default="/scratch-shared/tmp.pfRzLUwLmA/MSCOCO", type=str,
                        help="folder containing train2014/, val2014/ and annotations/")
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--save_every", default=10, type=int)
    parser.add_argument("--val_every", default=5, type=int)
    parser.add_argument("--logit_scale_init", default=None, type=float, help="initial logit scale (actual value, e.g. 100); stored as log in prior.logit_scale")
    parser.set_defaults(wandb_project="cs-aligner-coco", expname="cs-aligner-coco")
    return parser.parse_args()


def make_collate(image_processor):
    def collate(batch):
        images, captions = zip(*batch)
        return image_processor(images=list(images), return_tensors="pt").pixel_values, captions
    return collate


def main():
    args = parse_args()
    pl.seed_everything(args.seed, workers=True)
    save_path = os.path.join(args.save_dir, args.expname)
    os.makedirs(save_path, exist_ok=True)

    # one process per GPU (srun sets SLURM_LOCALID): build the backbone on this rank's GPU, not all on cuda:0
    device = "cpu"
    if torch.cuda.is_available():
        device = f"cuda:{int(os.environ.get('LOCAL_RANK', os.environ.get('SLURM_LOCALID', 0)))}"
        torch.cuda.set_device(device)
    backbone = CSAlignerBackbone(device, logit_scale_init=args.logit_scale_init, load_decoder=not args.no_decoder)
    collate = make_collate(backbone.image_processor)

    flip = transforms.RandomHorizontalFlip()
    train_set = CocoCaptions(os.path.join(args.coco_root, "train2014"),
                             os.path.join(args.coco_root, "annotations/captions_train2014.json"), transform=flip)
    val_set = CocoCaptions(os.path.join(args.coco_root, "val2014"),
                           os.path.join(args.coco_root, "annotations/captions_val2014.json"), transform=flip)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                              prefetch_factor=2, persistent_workers=True, pin_memory=True, collate_fn=collate)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, num_workers=args.num_workers,
                            pin_memory=True, collate_fn=collate)

    model = CSAligner(args, backbone)

    callbacks = [
        ModelCheckpoint(dirpath=save_path, monitor="val/val_loss", mode="min", save_top_k=1,
                        filename="cs-aligner-coco-best-{epoch:04d}"),
        ModelCheckpoint(dirpath=save_path, every_n_epochs=args.save_every, save_top_k=-1,
                        filename="cs-aligner-coco-periodic-{epoch:04d}"),
    ]
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        check_val_every_n_epoch=args.val_every,
        accelerator="gpu",
        devices=args.devices,
        strategy=DDPStrategy(find_unused_parameters=False, static_graph=True, process_group_backend="nccl"),
        precision="16-mixed",
        callbacks=callbacks,
        logger=WandbLogger(project=args.wandb_project, name=args.expname, config=vars(args), log_model=False),
    )
    trainer.fit(model, train_loader, val_loader, ckpt_path=args.resume)


if __name__ == "__main__":
    main()
