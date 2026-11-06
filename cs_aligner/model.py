"""CS-Aligner backbone: a trainable ECLIPSE prior on top of frozen Kandinsky 2.2.

Only ``self.prior`` (the ECLIPSE PriorTransformer, plus its ``logit_scale``) is
trained.  The CLIP ViT-bigG text / image encoders and the Kandinsky 2.2 decoder
stay frozen and are only used to produce targets and to decode images.
"""
import math

import torch
import torch.nn as nn
from diffusers import DiffusionPipeline
from transformers import CLIPProcessor, CLIPTextModelWithProjection, CLIPTokenizer

from cs_aligner import losses
from cs_aligner.prior.pipeline_kandinsky_prior import KandinskyPriorPipeline
from cs_aligner.prior.prior_transformer import PriorTransformer

CLIP_NAME = "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"
PRIOR_CONFIG = "ECLIPSE-Community/ECLIPSE_KandinskyV22_Prior"
KANDINSKY_PRIOR = "kandinsky-community/kandinsky-2-2-prior"
KANDINSKY_DECODER = "kandinsky-community/kandinsky-2-2-decoder"


class CSAlignerBackbone(nn.Module):
    def __init__(self, device, logit_scale_init=None, load_decoder=True):
        super().__init__()
        self.text_encoder = (
            CLIPTextModelWithProjection.from_pretrained(
                CLIP_NAME, projection_dim=1280, torch_dtype=torch.float16
            )
            .eval()
            .requires_grad_(False)
            .to(device)
        )
        self.image_processor = CLIPProcessor.from_pretrained(CLIP_NAME)
        self.tokenizer = CLIPTokenizer.from_pretrained(CLIP_NAME)

        # Trainable prior, initialised from the ECLIPSE config (random weights).
        self.prior = PriorTransformer(**PriorTransformer.load_config(PRIOR_CONFIG))
        # logit_scale_init is the actual scale (e.g. 100); the parameter is stored in log space
        if logit_scale_init is None:
            logit_scale_init = 1 / 0.07
        self.prior.logit_scale = nn.Parameter(torch.tensor(math.log(logit_scale_init)))
        self.prior.requires_grad_(True)

        self.pipe_prior = KandinskyPriorPipeline.from_pretrained(
            KANDINSKY_PRIOR,
            prior=self.prior,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            torch_dtype=torch.float16,
        ).to(device)
        self.pipe_prior.image_encoder = self.pipe_prior.image_encoder.eval().requires_grad_(False).to(device)
        self.image_encoder = self.pipe_prior.image_encoder

        # only needed to decode images (validation samples, FID, sample_t2i.py), not for training
        self.pipe = None
        if load_decoder:
            self.pipe = DiffusionPipeline.from_pretrained(KANDINSKY_DECODER, torch_dtype=torch.float16).to(device)

    # ------------------------------------------------------------------ losses
    def alignment_cosine(self, image_proj, text_proj, all_image=None, all_text=None, rank=0):
        return losses.info_nce(image_proj, text_proj, self.prior.logit_scale.exp(),
                               all_image=all_image, all_text=all_text, rank=rank)

    def alignment_csdiv(self, image_proj, text_proj, sigma=1.0):
        return losses.cs_divergence(image_proj, text_proj, sigma=sigma)

    # --------------------------------------------------------------- inference
    @torch.no_grad()
    def decode(self, image_embeds, negative_image_embeds=None, num_inference_steps=50, guidance_scale=7.5, **kwargs):
        """Decode (predicted) CLIP image embeddings into images with Kandinsky 2.2."""
        if negative_image_embeds is None:
            negative_image_embeds = self.pipe_prior.get_zero_embed(image_embeds.shape[0], device=image_embeds.device)
        return self.pipe(
            image_embeds=image_embeds,
            negative_image_embeds=negative_image_embeds,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            **kwargs,
        ).images


def load_prior_checkpoint(backbone, ckpt_path, strict=True, map_location="cpu"):
    """Load a CS-Aligner Lightning checkpoint (keys ``model.prior.*``) into ``backbone.prior``."""
    state_dict = torch.load(ckpt_path, map_location=map_location)["state_dict"]
    prior_sd = {k[len("model.prior."):]: v for k, v in state_dict.items() if k.startswith("model.prior.")}
    backbone.prior.load_state_dict(prior_sd, strict=strict)
    return backbone
