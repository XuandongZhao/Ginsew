import hashlib
from typing import List

import numpy as np
import torch
from transformers import LogitsWarper
import torch.nn.functional as F


class WatermarkBase:
    """
    Base class for watermarking distributions.

    Args:
        fraction: The fraction of the distribution to be green-listed.
        strength: The strength of the green-listing. Higher values result in higher logit scores for green-listed tokens.
        vocab_size: The size of the vocabulary.
        watermark_key: The random seed for the green-listing.
        freq: The frequency of the watermark.
        eps: The epsilon value for the watermark.
    """

    def __init__(self, fraction: float = 0.5, strength: float = 2.0, vocab_size: int = 50257, watermark_key: int = 0, freq: int = 16, eps: float = 0.2):
        rng = np.random.default_rng(self._hash_fn(watermark_key))
        mask = np.array([True] * int(fraction * vocab_size) + [False] * (vocab_size - int(fraction * vocab_size)))
        rng.shuffle(mask)
        self.vec = torch.tensor(rng.normal(loc=0, scale=1, size=(vocab_size, 256)))
        self.key = torch.tensor(rng.random(256))
        self.freq = freq
        self.eps = eps
        self.green_list_mask = torch.tensor(mask, dtype=torch.float32)
        self.strength = strength
        self.fraction = fraction

    @staticmethod
    def _hash_fn(x: int) -> int:
        """solution from https://stackoverflow.com/questions/67219691/python-hash-function-that-returns-32-or-64-bits"""
        x = np.int64(x)
        return int.from_bytes(hashlib.sha256(x).digest()[:4], 'little')


class WatermarkLogitsWarper(WatermarkBase, LogitsWarper):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.FloatTensor:
        """Add the watermark to the logits and return new logits."""
        x = torch.matmul(self.vec[input_ids[0][5]], self.key)
        x_ = torch.distributions.Normal(0, 1).cdf(x / np.sqrt(self.key.shape[0] / 3))
        probs = F.softmax(scores[0], dim=0)
        g0_idx = torch.where(self.green_list_mask == 0)[0]
        g1_idx = torch.where(self.green_list_mask == 1)[0]
        g0_prob = probs[g0_idx].sum()
        g1_prob = probs[g1_idx].sum()
        z0 = torch.cos(self.freq * x_)
        z1 = torch.cos(torch.add(self.freq * x_, torch.pi))
        new_g0_prob = (g0_prob + 1e-25 + self.eps * (1 + z0)) / (1 + 2 * self.eps + 1e-25)
        new_g1_prob = (g1_prob + 1e-25 + self.eps * (1 + z1)) / (1 + 2 * self.eps + 1e-25)
        new_probs = probs.clone()
        new_probs[g0_idx] = new_g0_prob / g0_prob * probs[g0_idx]
        new_probs[g1_idx] = new_g1_prob / g1_prob * probs[g1_idx]
        new_logits = torch.log(new_probs).view(1, -1)
        return new_logits
