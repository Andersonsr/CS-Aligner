"""Zero-shot classification of a CS-Aligner prior on the test sets of clip-refine.

The images go through the frozen CLIP ViT-bigG image encoder, the prompt "a photo of <class name>" of each class
through either
  - the trained prior (text -> predicted image embedding), for each checkpoint of --ckpt: run --name (default
    "CS-Aligner") with one checkpoint, "<experiment folder>/<checkpoint name>" with several
  - the CLIP ViT-bigG text encoder alone: run "CLIP ViT-bigG", the reference the prior is compared with
and an image gets the class of highest cosine similarity. The results are written as
{dataset: {run: {acc1, acc5, mean_per_class_recall}}}, the layout of clip-refine's merged results:
    python eval_zeroshot.py --data_root /path/to/datasets --ckpt save/<exp>/<checkpoint>.ckpt
    python eval_zeroshot.py --data_root /path/to/datasets --ckpt save/<exp>/*.ckpt save/<exp2>/*.ckpt
The images are encoded once per dataset and shared by all the checkpoints.
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
from cs_aligner.zeroshot_datasets import DATASETS, classname_path, load_dataset, read_classnames

CLIP_RUN = "CLIP ViT-bigG"


def parse_args():
    parser = argparse.ArgumentParser(description="CS-Aligner zero-shot classification")
    parser.add_argument("--data_root", required=True, type=str, help="folder given to download_datasets.sh --data")
    parser.add_argument("--ckpt", nargs="*", default=[], type=str,
                        help="CS-Aligner checkpoint(s); without any only CLIP is run")
    parser.add_argument("--name", default="CS-Aligner", type=str,
                        help="run name of the prior in the results, when a single checkpoint is given")
    parser.add_argument("--out", default=None, type=str,
                        help="results json (default <ckpt without .ckpt>_zeroshot.json with one checkpoint, "
                             "zeroshot.json next to the first checkpoint with several, zeroshot_clip.json without any)")
    parser.add_argument("--datasets", nargs="+", choices=list(DATASETS), default=list(DATASETS))
    parser.add_argument("--class_prompt", default="a photo of", type=str)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--seed", default=42, type=int, help="seed of the noise input of the prior")
    parser.add_argument("--num_samples", default=1, type=int,
                        help="noise samples of the prior per prompt; their normalised embeddings are averaged")
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
def prior_text_features(model, prompts, device, batch_size, seed, num_samples=1):
    generator = torch.Generator(device).manual_seed(seed)  # the prior starts from noise: same seed, same classifier
    feats = []
    for i in range(0, len(prompts), batch_size):
        emb = 0
        for _ in range(num_samples):  # mean of the normalised embeddings of num_samples noise draws
            with torch.autocast("cuda", dtype=torch.float16):
                sample = model.pipe_prior(prompts[i:i + batch_size], generator=generator).image_embeds
            emb = emb + F.normalize(sample.float(), dim=-1)
        feats.append(F.normalize(emb, dim=-1).cpu())
    return torch.cat(feats)


def classification_metrics(logits, target):
    # as clip-refine's metrics/zeroshot_classification.py: acc5 is nan with fewer than 5 classes
    pred = logits.argmax(dim=1)
    acc1 = (pred == target).float().mean().item()
    acc5 = float("nan")
    if logits.shape[1] >= 5:
        acc5 = (logits.topk(5, dim=1).indices == target[:, None]).any(dim=1).float().mean().item()
    return {"acc1": acc1, "acc5": acc5, "mean_per_class_recall": balanced_accuracy_score(target, pred)}


def run_name(ckpt):
    # save/<exp>/<checkpoint>.ckpt -> <exp>/<checkpoint>
    return f"{os.path.basename(os.path.dirname(os.path.abspath(ckpt)))}/{os.path.splitext(os.path.basename(ckpt))[0]}"


def runs_and_out(ckpts, name, out, kind):
    """{run name: checkpoint} and the results path: <ckpt>_<kind>.json with one checkpoint (run `name`),
    <kind>.json next to the first checkpoint with several, <kind>_clip.json without any; `out` overrides the path."""
    if len(ckpts) == 1:
        return {name: ckpts[0]}, out or os.path.splitext(ckpts[0])[0] + f"_{kind}.json"
    runs = {run_name(c): c for c in ckpts}
    if len(runs) != len(ckpts):
        raise ValueError("two checkpoints share the same <experiment folder>/<checkpoint name>")
    return runs, out or (os.path.join(os.path.dirname(ckpts[0]), f"{kind}.json") if runs else f"{kind}_clip.json")


def main():
    args = parse_args()
    runs, out = runs_and_out(args.ckpt, args.name, args.out, "zeroshot")
    device = "cuda"
    torch.manual_seed(args.seed)

    model = CSAlignerBackbone(device, load_decoder=False)
    model.eval()
    preprocess = Preprocess(model.image_processor)

    # text classifiers first, so each checkpoint is loaded once
    prompts = {name: [f"{args.class_prompt} {c}"
                      for c in read_classnames(classname_path(DATASETS[name][0].classname_file))]
               for name in args.datasets}
    classifiers = {name: {CLIP_RUN: clip_text_features(model, p, device, args.batch_size)} for name, p in prompts.items()}
    for run, ckpt in runs.items():
        print(f"## loading {ckpt} as {run}")
        load_prior_checkpoint(model, ckpt, strict=True, map_location=device)
        for name, p in prompts.items():
            classifiers[name][run] = prior_text_features(model, p, device, args.batch_size, args.seed,
                                                                 args.num_samples)

    results = {}
    for name in args.datasets:
        print(f"## {name}")
        dataset, _ = load_dataset(name, args.data_root, preprocess)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                            pin_memory=True)
        images, target = image_features(model, loader, device)
        results[name] = {run: classification_metrics(images @ text.T, target) for run, text in classifiers[name].items()}
        print(f"## {name}: {results[name]}")
        with open(out, "w") as f:  # after every dataset, so a job cut short keeps what it finished
            json.dump(results, f, indent=2)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
