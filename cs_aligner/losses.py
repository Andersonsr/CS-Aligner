"""Losses used by CS-Aligner.

* ``cs_divergence``     - Cauchy-Schwarz divergence between the image and text
                          embedding distributions (main alignment objective).
* ``info_nce``          - symmetric CLIP-style contrastive loss (auxiliary objective).

Parts of the kernel code are adapted from
https://github.com/yaohungt/Pointwise_Dependency_Neural_Estimation
"""
import torch
import torch.nn.functional as F


def gaussian_kernel(x, y, sigma):
    """Gaussian kernel matrix K[i, j] = exp(-||x_i - y_j||^2 / (2 sigma^2))."""
    xx = (x * x).sum(-1).unsqueeze(-1).repeat(1, y.shape[0])
    yy = (y * y).sum(-1).unsqueeze(-1).T.repeat(x.shape[0], 1)
    sq_dist = xx + yy - 2 * x @ y.T
    return torch.clamp(torch.exp(-sq_dist / 2 / sigma ** 2), min=0)


def cs_divergence(x1, x2, sigma=1.0):
    """Empirical Cauchy-Schwarz divergence D_CS(p(x1) || p(x2)).

    D_CS = -2 log E[k(x1, x2)] + log E[k(x1, x1')] + log E[k(x2, x2')]
    """
    k11 = gaussian_kernel(x1, x1, sigma)
    k22 = gaussian_kernel(x2, x2, sigma)
    k12 = gaussian_kernel(x1, x2, sigma)

    n1, n2 = k11.shape[0], k22.shape[0]
    self_term1 = k11.sum() / (n1 ** 2)
    self_term2 = k22.sum() / (n2 ** 2)
    cross_term = k12.sum() / (n1 * n2)

    return -2 * torch.log(cross_term) + torch.log(self_term1) + torch.log(self_term2)


def info_nce(image_proj, text_proj, logit_scale, all_image=None, all_text=None, rank=0):
    """Symmetric contrastive (CLIP) loss on L2-normalised embeddings.

    OpenCLIP-style distributed variant: rows are the local samples, columns are
    the features gathered from all ranks (``all_image`` / ``all_text``); the
    positive of local sample ``i`` sits at column ``rank * B + i``.
    """
    all_image = image_proj if all_image is None else all_image
    all_text = text_proj if all_text is None else all_text
    logits_per_image = logit_scale * image_proj @ all_text.t()
    logits_per_text = logit_scale * text_proj @ all_image.t()
    labels = torch.arange(len(image_proj), device=image_proj.device) + rank * len(image_proj)
    return (F.cross_entropy(logits_per_image, labels) + F.cross_entropy(logits_per_text, labels)) / 2
