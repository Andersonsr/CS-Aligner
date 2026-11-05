"""Build the MSCOCO FID statistics used during validation (--fid_stat).

The images of a split are resized (shorter side) and center cropped to --size, and the Inception statistics of the
crops are saved as an .npz with mu and sigma:
    python make_fid_stats.py --coco_root /path/to/coco --split val2017 --out fid_stats_mscoco256_val2017.npz
"""
import argparse
import os
import tempfile

from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from cs_aligner.metrics.fid_score import save_statistics_of_path


def main():
    parser = argparse.ArgumentParser(description="MSCOCO FID statistics")
    parser.add_argument("--coco_root", required=True, type=str, help="folder containing <split>/")
    parser.add_argument("--split", default="val2017", type=str)
    parser.add_argument("--out", required=True, type=str, help="output .npz")
    parser.add_argument("--size", default=256, type=int)
    parser.add_argument("--num_workers", default=8, type=int)
    args = parser.parse_args()

    crop = transforms.Compose([
        transforms.Resize(args.size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(args.size),
    ])
    src = os.path.join(args.coco_root, args.split)
    with tempfile.TemporaryDirectory() as tmp:
        for name in tqdm(sorted(os.listdir(src)), desc=f"resizing {args.split}"):
            img = crop(Image.open(os.path.join(src, name)).convert("RGB"))
            img.save(os.path.join(tmp, os.path.splitext(name)[0] + ".png"))
        save_statistics_of_path(tmp, args.out, num_workers=args.num_workers)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
