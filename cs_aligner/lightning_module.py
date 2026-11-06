"""PyTorch-Lightning training module shared by the COCO and LAION experiments.

Training objective (on L2-normalised embeddings):

    loss = D_CS(img, text)                  # Cauchy-Schwarz divergence
         + aux_weight * InfoNCE(img, text)  # contrastive auxiliary loss
"""
import math
import os
import random
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.optim as optim
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities import rank_zero_only
from sklearn.manifold import TSNE
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm

from cs_aligner.metrics.fid_score import calculate_fid_given_paths


def add_common_args(parser):
    """Arguments shared by train_coco.py and train_laion.py."""
    parser.add_argument("--devices", default=1, type=int)
    parser.add_argument("--epochs", default=1000, type=int)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--lr", default=5e-5, type=float)
    parser.add_argument("--save_dir", default="./save", type=str)
    parser.add_argument("--expname", default="cs-aligner", type=str)
    parser.add_argument("--wandb_project", default="cs-aligner", type=str)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--resume", default=None, type=str, help="path to a Lightning checkpoint to resume from")

    # objective
    parser.add_argument("--alignment_type", default="csdiv", choices=["csdiv", "cosine"],
                        help="csdiv = CS-Aligner; cosine = InfoNCE-only baseline")
    parser.add_argument("--aux_weight", default=0.1, type=float, help="weight of the InfoNCE auxiliary loss")
    parser.add_argument("--csdiv_sigma", default=1.0, type=float, help="Gaussian kernel width of D_CS")
    parser.add_argument("--negative_prompt", default="", type=str,
                        help='negative prompt for the prior, e.g. "low quality, bad quality"; empty = none')
    parser.add_argument("--logit_scale_schedule", action="store_true",
                        help="cosine-schedule the logit scale from --logit_scale_init to --logit_scale_final "
                             "instead of learning it")
    parser.add_argument("--logit_scale_final", default=None, type=float,
                        help="final logit scale (actual value) of the cosine schedule")
    parser.add_argument("--logit_scale_schedule_frac", default=0.75, type=float,
                        help="fraction of the total training steps over which the logit scale reaches its final value")

    # evaluation during training
    parser.add_argument("--fid_stat", type=str,
                        default="/home/wenzhey/path-to-project/CodeBase/Generation/fid_stats/fid_stats_mscoco256_val.npz")
    parser.add_argument("--fid_every", default=10, type=int, help="compute FID every N epochs")
    parser.add_argument("--fid_batches", default=15, type=int, help="number of decode batches used for FID")
    parser.add_argument("--decode_batch_size", default=32, type=int)
    parser.add_argument("--log_every", default=50, type=int)
    parser.add_argument("--no_decoder", action="store_true",
                        help="do not load the Kandinsky decoder: no sample grid and no FID during validation")
    return parser


def pick_caption(captions):
    """COCO gives a list of captions per image -> sample one at random."""
    if isinstance(captions, (tuple, list)):
        return [random.choice(c) if isinstance(c, (list, tuple)) else c for c in captions]
    return captions


class CSAligner(pl.LightningModule):
    def __init__(self, args, backbone, max_val_samples=1000):
        super().__init__()
        self.args = args
        self.model_warp = backbone  # attribute name kept for checkpoint compatibility
        self.max_val_samples = max_val_samples
        self._val_img, self._val_txt, self._val_neg = [], [], []

        if args.logit_scale_schedule:
            if args.logit_scale_final is None:
                raise ValueError("--logit_scale_schedule requires --logit_scale_final")
            # the scale is set by the schedule, not learned
            self.model_warp.prior.logit_scale.requires_grad_(False)

    # Only the prior is saved / restored (keys: "model.prior.*").
    def state_dict(self, *args, **kwargs):
        return {"model.prior." + k: v for k, v in self.model_warp.prior.state_dict().items()}

    def load_state_dict(self, state_dict, strict=False, **kwargs):
        prior_sd = {k[len("model.prior."):]: v for k, v in state_dict.items() if k.startswith("model.prior.")}
        return self.model_warp.prior.load_state_dict(prior_sd, strict=strict)

    # ------------------------------------------------------------------ core
    def _negative_prompt(self, n):
        return [self.args.negative_prompt] * n if self.args.negative_prompt else None

    def _embed(self, images, captions):
        captions = pick_caption(captions)
        text_embs, neg_embs = self.model_warp.pipe_prior(
            captions, negative_prompt=self._negative_prompt(len(captions))
        ).to_tuple()
        with torch.no_grad():
            img_embs, _ = self.model_warp.image_encoder(images).to_tuple()
        return img_embs.detach(), text_embs, neg_embs

    def _losses(self, img_proj, text_proj, gather=False):
        # OpenCLIP-style: gather features from all ranks with gradients. InfoNCE uses
        # local rows x global columns; CS divergence (log of global kernel means, not
        # row-decomposable) uses the full global batch. In both cases the summed
        # gradients across ranks, averaged by DDP, equal the global-batch gradient.
        all_img, all_txt, rank = img_proj, text_proj, 0
        if gather and self.trainer.world_size > 1:
            all_img = self.all_gather(img_proj, sync_grads=True).flatten(0, 1)
            all_txt = self.all_gather(text_proj, sync_grads=True).flatten(0, 1)
            rank = self.global_rank

        if self.args.alignment_type == "csdiv":
            align = self.model_warp.alignment_csdiv(all_img, all_txt, sigma=self.args.csdiv_sigma)
            aux = self.model_warp.alignment_cosine(img_proj, text_proj, all_img, all_txt, rank)
        else:  # InfoNCE-only baseline
            align = self.model_warp.alignment_cosine(img_proj, text_proj, all_img, all_txt, rank)
            aux = torch.zeros((), device=img_proj.device)
        return align, aux

    def on_train_start(self):
        if self.args.logit_scale_schedule:
            total = self.trainer.estimated_stepping_batches
            if math.isinf(total):  # WebDataset loader has no length: use its fixed epoch size (train_laion.py)
                total = self.trainer.max_epochs * (self.args.samples_per_epoch // self.args.batch_size)
            self._logit_scale_steps = max(1, int(self.args.logit_scale_schedule_frac * total))

    def on_train_batch_start(self, batch, batch_idx):
        # cosine from the initial to the final scale over _logit_scale_steps, then constant
        if self.args.logit_scale_schedule:
            start = self.args.logit_scale_init if self.args.logit_scale_init is not None else 1 / 0.07
            end = self.args.logit_scale_final
            t = min(self.global_step / self._logit_scale_steps, 1.0)
            scale = end + 0.5 * (start - end) * (1 + math.cos(math.pi * t))
            with torch.no_grad():
                self.model_warp.prior.logit_scale.fill_(math.log(scale))

    def training_step(self, batch, batch_idx):
        images, captions = batch
        img_embs, text_embs, _ = self._embed(images, captions)

        img_proj = F.normalize(img_embs, dim=-1)
        text_proj = F.normalize(text_embs, dim=-1)
        align, aux = self._losses(img_proj, text_proj, gather=True)
        loss = align + self.args.aux_weight * aux

        if batch_idx % self.args.log_every == 0:
            logit_scale = self.model_warp.prior.logit_scale.detach().exp()
            self.log_dict({"train/train_loss": loss, "train/align_loss": align,
                           "train/aux_loss": aux, "train/logit_scale": logit_scale})
            if self.trainer.is_global_zero:
                print(f"[train] epoch={self.current_epoch} step={batch_idx} loss={loss.item():.4f} "
                      f"align={align.item():.4f} aux={float(aux):.4f} logit_scale={logit_scale.item():.2f}")
        return loss

    def validation_step(self, batch, batch_idx):
        images, captions = batch
        img_embs, text_embs, neg_embs = self._embed(images, captions)

        if sum(len(x) for x in self._val_img) < self.max_val_samples:
            self._val_img.append(img_embs)
            self._val_txt.append(text_embs)
            self._val_neg.append(neg_embs)

        img_proj = F.normalize(img_embs, dim=-1)
        text_proj = F.normalize(text_embs, dim=-1)
        align, aux = self._losses(img_proj, text_proj)
        loss = align + self.args.aux_weight * aux

        self.log_dict({"val/val_loss": loss, "val/align_loss": align, "val/aux_loss": aux},
                      on_step=False, on_epoch=True, sync_dist=True)
        return loss

    # ----------------------------------------------------------- visualisation
    @rank_zero_only
    def _log_wandb(self, data):
        if isinstance(self.logger, WandbLogger):
            import wandb
            self.logger.experiment.log({k: wandb.Image(v) for k, v in data.items()})

    def on_validation_epoch_end(self):
        if not self._val_img:
            return
        n = self.max_val_samples
        img_feats = torch.cat(self._val_img)[:n]
        txt_feats = torch.cat(self._val_txt)[:n]
        neg_feats = torch.cat(self._val_neg)[:n]
        self._val_img, self._val_txt, self._val_neg = [], [], []

        # 1) sample grid: top row = from predicted (text->image) embeddings, bottom row = from GT image embeddings
        if not self.args.no_decoder:
            to_tensor = transforms.ToTensor()
            pred = self.model_warp.decode(txt_feats[-4:], neg_feats[-4:])
            gt = self.model_warp.decode(img_feats[-4:])
            grid = make_grid(torch.stack([to_tensor(im) for im in pred + gt]), nrow=4)
            self._log_wandb({"samples": grid})

        # 2) FID of images decoded from predicted embeddings vs. MSCOCO val statistics
        if not self.args.no_decoder and (self.current_epoch + 1) % self.args.fid_every == 0:
            bs = self.args.decode_batch_size
            with tempfile.TemporaryDirectory() as tmp:
                idx = 0
                for i in tqdm(range(self.args.fid_batches), desc="FID decoding"):
                    emb, neg = txt_feats[i * bs:(i + 1) * bs], neg_feats[i * bs:(i + 1) * bs]
                    if emb.shape[0] == 0:
                        break
                    for im in self.model_warp.decode(emb, neg):
                        im.save(os.path.join(tmp, f"{idx}.png"))
                        idx += 1
                fid = calculate_fid_given_paths((self.args.fid_stat, tmp))
            self.log("fid_score", float(fid))

        # 3) t-SNE of normalised image vs. text embeddings
        feats = torch.cat([F.normalize(img_feats, dim=-1), F.normalize(txt_feats, dim=-1)]).float().cpu().numpy()
        tsne = TSNE(n_components=2, random_state=42).fit_transform(feats)
        k = img_feats.shape[0]
        fig = plt.figure(figsize=(8, 8))
        plt.scatter(tsne[:k, 0], tsne[:k, 1], color="r", label="Image Features")
        plt.scatter(tsne[k:, 0], tsne[k:, 1], color="b", label="Text Features")
        plt.legend(loc="best")
        plt.title("t-SNE of Image and Text Features")
        self._log_wandb({"t-SNE Plot": fig})
        plt.close(fig)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.model_warp.prior.parameters(), lr=self.args.lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args.epochs)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
