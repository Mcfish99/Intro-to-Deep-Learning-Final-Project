import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim_fn

def compute_mse(x_real, x_pred):
    return torch.mean((x_real - x_pred) ** 2).item()

def compute_ssim(x_real, x_pred):
    x_real_np = ((x_real.detach().cpu().numpy() + 1) / 2).clip(0, 1)
    x_pred_np = ((x_pred.detach().cpu().numpy() + 1) / 2).clip(0, 1)
    scores = []
    for r, p in zip(x_real_np, x_pred_np):
        r = r.transpose(1, 2, 0)
        p = p.transpose(1, 2, 0)
        scores.append(ssim_fn(r, p, channel_axis=-1, data_range=1.0))
    return float(np.mean(scores))

def compute_lpips(x_real, x_pred, lpips_fn):
    with torch.no_grad():
        score = lpips_fn(x_real, x_pred)
    return score.mean().item()

def compute_fid(real_dir, fake_dir, device='cpu'):
    from pytorch_fid import fid_score
    return fid_score.calculate_fid_given_paths(
        [real_dir, fake_dir], batch_size=50, device=device, dims=2048
    )

def evaluate_all(x_real, x_pred, lpips_fn=None):
    results = {
        'mse':  compute_mse(x_real, x_pred),
        'ssim': compute_ssim(x_real, x_pred),
    }
    if lpips_fn is not None:
        results['lpips'] = compute_lpips(x_real, x_pred, lpips_fn)
    return results
