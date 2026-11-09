"""Image-text retrieval of a CS-Aligner prior on MSCOCO val2017 (5000 images, ~25k captions).

The images go through the frozen CLIP ViT-bigG image encoder, the captions through either
  - the trained prior (text -> predicted image embedding), for each checkpoint of --ckpt: run --name (default
    "CS-Aligner") with one checkpoint, "<experiment folder>/<checkpoint name>" with several
  - the CLIP ViT-bigG text encoder alone: run "CLIP ViT-bigG", the reference the prior is compared with
and are ranked by cosine similarity. Recall@k, as in clip_benchmark:
  - text_retrieval_recall@k (image -> text): an image is a hit if any of its captions is in its top k
  - image_retrieval_recall@k (text -> image): a caption is a hit if its image is in its top k
The results are written as {"retrieval_coco_<split>": {run: {metric: value}}}:
    python eval_retrieval.py --coco_root /path/to/coco --ckpt save/<exp>/*.ckpt
coco_root holds <split>/ and annotations/captions_<split>.json.
"""
import argparse
import json
import os

import torch
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import DataLoader, Dataset

from cs_aligner.model import CSAlignerBackbone, load_prior_checkpoint
from eval_zeroshot import (CLIP_RUN, Preprocess, clip_text_features, image_features, prior_text_features,
                           runs_and_out)

KS = (1, 5, 10)


def parse_args():
    parser = argparse.ArgumentParser(description="CS-Aligner image-text retrieval on MSCOCO")
    parser.add_argument("--coco_root", required=True, type=str, help="folder with <split>/ and annotations/")
    parser.add_argument("--split", default="val2017", type=str)
    parser.add_argument("--ckpt", nargs="*", default=[], type=str,
                        help="CS-Aligner checkpoint(s); without any only CLIP is run")
    parser.add_argument("--name", default="CS-Aligner", type=str,
                        help="run name of the prior in the results, when a single checkpoint is given")
    parser.add_argument("--out", default=None, type=str,
                        help="results json (default <ckpt without .ckpt>_retrieval.json with one checkpoint, "
                             "retrieval.json next to the first checkpoint with several, "
                             "retrieval_clip.json without any)")
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--seed", default=42, type=int, help="seed of the noise input of the prior")
    return parser.parse_args()


class CocoImages(Dataset):
    # (pixel values, image index); captions[j] belongs to image caption_image[j]
    def __init__(self, root, ann_file, transform):
        coco = COCO(ann_file)
        self.ids = sorted(coco.imgs)
        self.paths = [os.path.join(root, coco.imgs[i]["file_name"]) for i in self.ids]
        self.captions, self.caption_image = [], []
        for idx, img_id in enumerate(self.ids):
            for ann in coco.imgToAnns[img_id]:
                self.captions.append(ann["caption"].strip())
                self.caption_image.append(idx)
        self.caption_image = torch.tensor(self.caption_image)
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        return self.transform(Image.open(self.paths[idx]).convert("RGB")), idx


def retrieval_metrics(images, texts, caption_image):
    # images (N, d) and texts (M, d) L2-normalised; caption_image (M,) = image index of each caption
    sim = images @ texts.T  # (N, M)
    k = max(KS)
    # text -> image: whether the caption's own image is among its top k images
    t2i = (sim.T.topk(min(k, len(images)), dim=1).indices == caption_image[:, None])  # (M, k)
    # image -> text: whether a caption of the image is among its top k captions
    i2t = (caption_image[sim.topk(min(k, len(texts)), dim=1).indices] == torch.arange(len(images))[:, None])  # (N, k)
    metrics = {}
    for kk in KS:
        metrics[f"image_retrieval_recall@{kk}"] = t2i[:, :kk].any(dim=1).float().mean().item()
        metrics[f"text_retrieval_recall@{kk}"] = i2t[:, :kk].any(dim=1).float().mean().item()
    return metrics


def main():
    args = parse_args()
    runs, out = runs_and_out(args.ckpt, args.name, args.out, "retrieval")
    device = "cuda"
    torch.manual_seed(args.seed)

    model = CSAlignerBackbone(device, load_decoder=False)
    model.eval()
    dataset = CocoImages(os.path.join(args.coco_root, args.split),
                         os.path.join(args.coco_root, "annotations", f"captions_{args.split}.json"),
                         Preprocess(model.image_processor))
    print(f"## {args.split}: {len(dataset)} images, {len(dataset.captions)} captions")

    texts = {CLIP_RUN: clip_text_features(model, dataset.captions, device, args.batch_size)}
    for run, ckpt in runs.items():
        print(f"## loading {ckpt} as {run}")
        load_prior_checkpoint(model, ckpt, strict=True, map_location=device)
        texts[run] = prior_text_features(model, dataset.captions, device, args.batch_size, args.seed)

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                        pin_memory=True)
    images, order = image_features(model, loader, device)
    assert torch.equal(order, torch.arange(len(dataset)))

    name = f"retrieval_coco_{args.split}"
    results = {name: {run: retrieval_metrics(images, t, dataset.caption_image) for run, t in texts.items()}}
    for run, m in results[name].items():
        print(f"## {run}: {m}")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
