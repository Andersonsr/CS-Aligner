"""Zero-shot classification of a CS-Aligner prior on the test sets of clip-refine.

The images go through the frozen CLIP ViT-bigG image encoder, the prompt "a photo of <class name>" of each class
through either
  - the trained prior (text -> predicted image embedding): run --name (default "CS-Aligner"), with --ckpt
  - the CLIP ViT-bigG text encoder alone: run "CLIP ViT-bigG", the reference the prior is compared with
and an image gets the class of highest cosine similarity. The results are written as
{dataset: {run: {acc1, acc5, mean_per_class_recall}}}, the layout of clip-refine's merged results:
    python eval_zeroshot.py --data_root /path/to/datasets --ckpt save/<exp>/<checkpoint>.ckpt
data_root is the folder given to vlm-tuning's download_datasets.sh --data (see cs_aligner/zeroshot_datasets.py).
"""
import argparse
import json
import os

import torch
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from cs_aligner.model import CSAlignerBackbone, load_prior_checkpoint
from cs_aligner.zeroshot_datasets import DATASETS, load_dataset, read_classnames

CLIP_RUN = "CLIP ViT-bigG"


def parse_args():
    parser = argparse.ArgumentParser(description="CS-Aligner zero-shot classification")
    parser.add_argument("--data_root", required=True, type=str, help="folder given to download_datasets.sh --data")
    parser.add_argument("--ckpt", default=None, type=str, help="CS-Aligner checkpoint; without it only CLIP is run")
    parser.add_argument("--name", default="CS-Aligner", type=str, help="run name of the prior in the results")
    parser.add_argument("--out", default=None, type=str,
                        help="results json (default <ckpt without .ckpt>_zeroshot.json, zeroshot_clip.json without --ckpt)")
    parser.add_argument("--datasets", nargs="+", choices=list(DATASETS), default=list(DATASETS))
    parser.add_argument("--class_prompt", default="a photo of", type=str)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--seed", default=42, type=int, help="seed of the noise input of the prior")
    return parser.parse_args()


class Preprocess:
    # PIL image -> pixel values of the CLIP processor (a class, so dataloader workers can pickle it)
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, image):
        return self.processor(images=image, return_tensors="pt").pixel_values[0]


@torch.no_grad()
def image_features(model, loader, device):
    feats, labels = [], []
    for images, target in tqdm(loader, desc="images"):
        with torch.autocast("cuda", dtype=torch.float16):
            emb = model.image_encoder(images.to(device)).image_embeds
        feats.append(F.normalize(emb.float(), dim=-1).cpu())
        labels.append(target)
    return torch.cat(feats), torch.cat(labels)


@torch.no_grad()
def clip_text_features(model, prompts, device, batch_size):
    feats = []
    for i in range(0, len(prompts), batch_size):
        tokens = model.tokenizer(prompts[i:i + batch_size], padding="max_length", truncation=True,
                                 max_length=model.tokenizer.model_max_length, return_tensors="pt").to(device)
        with torch.autocast("cuda", dtype=torch.float16):
            emb = model.text_encoder(**tokens).text_embeds
        feats.append(F.normalize(emb.float(), dim=-1).cpu())
    return torch.cat(feats)


@torch.no_grad()
def prior_text_features(model, prompts, device, batch_size, seed):
    generator = torch.Generator(device).manual_seed(seed)  # the prior starts from noise: same seed, same classifier
    feats = []
    for i in range(0, len(prompts), batch_size):
        with torch.autocast("cuda", dtype=torch.float16):
            emb = model.pipe_prior(prompts[i:i + batch_size], generator=generator).image_embeds
        feats.append(F.normalize(emb.float(), dim=-1).cpu())
    return torch.cat(feats)


def classification_metrics(logits, target):
    # as clip-refine's metrics/zeroshot_classification.py: acc5 is nan with fewer than 5 classes
    pred = logits.argmax(dim=1)
    acc1 = (pred == target).float().mean().item()
    acc5 = float("nan")
    if logits.shape[1] >= 5:
        acc5 = (logits.topk(5, dim=1).indices == target[:, None]).any(dim=1).float().mean().item()
    return {"acc1": acc1, "acc5": acc5, "mean_per_class_recall": balanced_accuracy_score(target, pred)}


def main():
    args = parse_args()
    out = args.out or (os.path.splitext(args.ckpt)[0] + "_zeroshot.json" if args.ckpt else "zeroshot_clip.json")
    device = "cuda"
    torch.manual_seed(args.seed)

    model = CSAlignerBackbone(device, load_decoder=False)
    if args.ckpt:
        load_prior_checkpoint(model, args.ckpt, strict=True, map_location=device)
    model.eval()
    preprocess = Preprocess(model.image_processor)

    results = {}
    for name in args.datasets:
        print(f"## {name}")
        dataset, classname_file = load_dataset(name, args.data_root, preprocess)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                            pin_memory=True)
        images, target = image_features(model, loader, device)

        prompts = [f"{args.class_prompt} {c}" for c in read_classnames(classname_file)]
        classifiers = {CLIP_RUN: clip_text_features(model, prompts, device, args.batch_size)}
        if args.ckpt:
            classifiers[args.name] = prior_text_features(model, prompts, device, args.batch_size, args.seed)

        results[name] = {run: classification_metrics(images @ text.T, target) for run, text in classifiers.items()}
        print(f"## {name}: {results[name]}")
        with open(out, "w") as f:  # after every dataset, so a job cut short keeps what it finished
            json.dump(results, f, indent=2)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
