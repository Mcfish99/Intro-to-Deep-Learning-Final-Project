import torch
import random

MASK_CONDITIONS = {
    'small':  (0.1, 0.2),
    'medium': (0.3, 0.4),
    'large':  (0.5, 0.6),
}

def generate_mask(H, W, condition='medium'):
    lo, hi = MASK_CONDITIONS[condition]
    h = random.randint(int(lo * H), int(hi * H))
    w = random.randint(int(lo * W), int(hi * W))
    cx = random.randint(w // 2, W - w // 2)
    cy = random.randint(h // 2, H - h // 2)
    mask = torch.zeros(1, H, W)
    mask[:, cy - h//2 : cy + h//2, cx - w//2 : cx + w//2] = 1
    return mask

def apply_mask(x, mask):
    return x * (1 - mask)

def batch_masks(batch_size, H, W, condition='medium'):
    return torch.stack([generate_mask(H, W, condition) for _ in range(batch_size)])
