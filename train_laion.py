"""Train CS-Aligner on LAION (high-resolution subset, WebDataset shards), validate on MSCOCO val2014.

Launch (see scripts/run_laion.sh):
    python train_laion.py --devices 1 --batch_size 1024 ...
"""
import argparse
import os

import pytorch_lightning as pl
import torch
import webdataset as wds
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from torch.utils.data import DataLoader
from torchvision.datasets import CocoCaptions

from cs_aligner.lightning_module import CSAligner, add_common_args
from cs_aligner.model import CSAlignerBackbone


def parse_args():
    parser = argparse.ArgumentParser(description="CS-Aligner on LAION")
    add_common_args(parser)
    parser.add_argument("--laion_shards", type=str,
                        default="/home/wenzhey/path-to-scratch/Data/laion-high-resolution-output/{00000..00389}.tar")
    parser.add_argument("--laion_cache", type=str, default=None)
    parser.add_argument("--samples_per_epoch", type=int, default=5_000_000)
    parser.add_argument("--min_image_size", type=int, default=100)
    parser.add_argument("--coco_root", default="/scratch-shared/tmp.pfRzLUwLmA/MSCOCO", type=str,
                        help="MSCOCO folder used for validation (val2014/ and annotations/)")
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--save_every", default=2, type=int)
    parser.add_argument("--val_every", default=2, type=int)
    parser.add_argument("--val_split", default="val2014", type=str, help="MSCOCO split to use for validation")
    parser.set_defaults(wandb_project="cs-aligner-laion", expname="cs-aligner-laion", fid_every=2)
    return parser.parse_args()


def make_laion_webdataset(urls, image_processor, batch_size, cache_dir=None, min_size=100,
                          image_key="jpg", caption_key="txt"):
    """Resampled WebDataset yielding batches of (pixel_values, captions)."""

    def has_keys(sample):
        return image_key in sample and caption_key in sample

    def big_enough(sample):
        img = sample[image_key]
        return img.width >= min_size and img.height >= min_size

    def preprocess(sample):
        pixels = image_processor(images=sample[image_key], return_tensors="pt").pixel_values.squeeze(0)
        caption = sample[caption_key]
        caption = caption.decode("utf-8") if isinstance(caption, bytes) else caption
        return pixels, caption

    return (
        wds.WebDataset(urls, resampled=True, cache_dir=cache_dir, nodesplitter=wds.split_by_node,
                       handler=wds.handlers.warn_and_continue)
        .decode("pil", handler=wds.handlers.warn_and_continue)
        .select(has_keys)
        .select(big_enough)
        .map(preprocess, handler=wds.handlers.warn_and_continue)
        .shuffle(2048)
        .batched(batch_size)
    )


class LaionDataModule(pl.LightningDataModule):
    def __init__(self, args, image_processor):
        super().__init__()
        self.args = args
        self.image_processor = image_processor
        self.val_split = args.val_split

    def setup(self, stage=None):
        a = self.args
        self.train_set = make_laion_webdataset(a.laion_shards, self.image_processor, a.batch_size,
                                               cache_dir=a.laion_cache, min_size=a.min_image_size)
        self.val_set = CocoCaptions(os.path.join(a.coco_root, self.val_split),
                                    os.path.join(a.coco_root, f"annotations/captions_{self.val_split}.json"))

    def train_dataloader(self):
        loader = wds.WebLoader(self.train_set, batch_size=None, num_workers=self.args.num_workers,
                               pin_memory=True, persistent_workers=True)
        return loader.with_epoch(self.args.samples_per_epoch // self.args.batch_size)

    def val_dataloader(self):
        def collate(batch):
            images, captions = zip(*batch)
            return self.image_processor(images=list(images), return_tensors="pt").pixel_values, captions
        return DataLoader(self.val_set, batch_size=self.args.batch_size, num_workers=4, shuffle=False,
                          collate_fn=collate)


def main():
    args = parse_args()
    pl.seed_everything(args.seed, workers=True)
    save_path = os.path.join(args.save_dir, args.expname)
    os.makedirs(save_path, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    backbone = CSAlignerBackbone(device, load_decoder=not args.no_decoder)
    model = CSAligner(args, backbone)
    datamodule = LaionDataModule(args, backbone.image_processor)

    callbacks = [
        ModelCheckpoint(dirpath=save_path, monitor="val/val_loss", mode="min", save_top_k=1,
                        filename="cs-aligner-laion-best-{epoch:04d}"),
        ModelCheckpoint(dirpath=save_path, every_n_epochs=args.save_every, save_top_k=-1,
                        filename="cs-aligner-laion-periodic-{epoch:04d}"),
    ]
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        check_val_every_n_epoch=args.val_every,
        accelerator="gpu",
        devices=args.devices,
        strategy=DDPStrategy(find_unused_parameters=True),
        precision="16-mixed",
        callbacks=callbacks,
        logger=WandbLogger(project=args.wandb_project, name=args.expname, config=vars(args), log_model=False),
    )
    trainer.fit(model, datamodule=datamodule, ckpt_path=args.resume)


if __name__ == "__main__":
    main()
