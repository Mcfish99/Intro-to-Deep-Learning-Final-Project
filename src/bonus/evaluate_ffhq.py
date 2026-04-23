import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import argparse
import sys
import csv
import torch
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import lpips as lpips_lib

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.shared.dataset  import FFHQDataset
from src.shared.mask     import batch_masks
from src.shared.evaluate import compute_mse, compute_ssim, compute_lpips, compute_fid

from src.reconstruction.model import UNetInpainting as ReconModel

# ---------------------------------------------------------------------------
# TODO: Replace these imports once your teammates finish their models.
# ---------------------------------------------------------------------------
# from src.gan.model  import GANGenerator   as GANModel
# from src.cvae.model import CVAEGenerator  as CVAEModel
# ---------------------------------------------------------------------------


def save_images(tensor, output_dir, prefix, start_idx):
    """Save a batch of images ([-1,1] float tensors) as PNG files."""
    os.makedirs(output_dir, exist_ok=True)
    imgs = (tensor.clamp(-1, 1) + 1) / 2   # -> [0, 1]
    for i, img in enumerate(imgs):
        path = os.path.join(output_dir, f"{prefix}_{start_idx + i:05d}.png")
        vutils.save_image(img, path)


def evaluate_model(model, loader, device, lpips_fn,
                   fake_dir, real_dir, condition='medium'):
    """
    Run inference on the full dataloader and collect per-batch metrics.
    Saves generated images to fake_dir and real images to real_dir for FID.
    Returns averaged MSE, SSIM, LPIPS (FID computed separately at the end).
    """
    model.eval()
    mse_scores, ssim_scores, lpips_scores = [], [], []
    img_idx = 0

    with torch.no_grad():
        for x in loader:
            x = x.to(device)
            B, C, H, W = x.shape
            mask  = batch_masks(B, H, W, condition=condition).to(device)
            y     = x * (1 - mask)
            x_hat = model(y, mask)

            mse_scores.append(compute_mse(x, x_hat))
            ssim_scores.append(compute_ssim(x, x_hat))
            lpips_scores.append(compute_lpips(x, x_hat, lpips_fn))

            # Save images for FID computation
            save_images(x_hat, fake_dir, 'fake', img_idx)
            save_images(x,     real_dir, 'real', img_idx)
            img_idx += B

    avg_mse   = sum(mse_scores)   / len(mse_scores)
    avg_ssim  = sum(ssim_scores)  / len(ssim_scores)
    avg_lpips = sum(lpips_scores) / len(lpips_scores)

    return avg_mse, avg_ssim, avg_lpips


def load_model(model_class, ckpt_path, device):
    model = model_class().to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    print(f"  Loaded checkpoint: {ckpt_path}")
    return model


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Load FFHQ dataset
    print("\nLoading FFHQ dataset...")
    dataset = FFHQDataset(
        data_dir   = args.ffhq_dir,
        image_size = args.image_size,
        download   = args.download_ffhq,
        max_images = args.max_images,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size,
                        shuffle=True, num_workers=0)
    print(f"  {len(dataset)} images loaded.")

    # 2. LPIPS function (shared across all models)
    lpips_fn = lpips_lib.LPIPS(net='vgg').to(device)

    # 3. Define models to evaluate
    models_cfg = [
        ('Reconstruction', ReconModel,  args.recon_ckpt),
        # ('GAN',           GANModel,    args.gan_ckpt),
        # ('CVAE',          CVAEModel,   args.cvae_ckpt),
    ]

    os.makedirs(args.output_dir, exist_ok=True)
    results = []

    real_dir = os.path.join(args.output_dir, 'real_images')

    for model_name, model_class, ckpt_path in models_cfg:
        print(f"\nEvaluating: {model_name}")

        if not os.path.exists(ckpt_path):
            print(f"  WARNING: checkpoint not found at '{ckpt_path}', skipping.")
            continue

        model    = load_model(model_class, ckpt_path, device)
        fake_dir = os.path.join(args.output_dir, f'fake_{model_name.lower()}')

        avg_mse, avg_ssim, avg_lpips = evaluate_model(
            model, loader, device, lpips_fn,
            fake_dir=fake_dir,
            real_dir=real_dir,
            condition='medium',
        )

        print(f"  Computing FID...")
        fid = compute_fid(real_dir, fake_dir, device=str(device))

        print(f"  MSE:   {avg_mse:.4f}")
        print(f"  SSIM:  {avg_ssim:.4f}")
        print(f"  LPIPS: {avg_lpips:.4f}")
        print(f"  FID:   {fid:.2f}")

        results.append({
            'Model': model_name,
            'MSE':   round(avg_mse,   4),
            'SSIM':  round(avg_ssim,  4),
            'LPIPS': round(avg_lpips, 4),
            'FID':   round(fid,       2),
        })

    # 4. Save results to CSV
    csv_path = os.path.join(args.output_dir, 'ffhq_results.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['Model', 'MSE', 'SSIM', 'LPIPS', 'FID'])
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to: {csv_path}")
    print("\n--- Summary ---")
    for r in results:
        print(f"  {r['Model']:15s}  MSE={r['MSE']}  SSIM={r['SSIM']}  "
              f"LPIPS={r['LPIPS']}  FID={r['FID']}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Bonus: FFHQ zero-shot evaluation')
    parser.add_argument('--ffhq_dir',      default='data/ffhq',
                        help='Directory to store/load FFHQ images')
    parser.add_argument('--recon_ckpt',    default='models/reconstruction_best.pth')
    parser.add_argument('--gan_ckpt',      default='models/gan_best.pth')
    parser.add_argument('--cvae_ckpt',     default='models/cvae_best.pth')
    parser.add_argument('--output_dir',    default='results/bonus')
    parser.add_argument('--batch_size',    type=int, default=8)
    parser.add_argument('--image_size',    type=int, default=256)
    parser.add_argument('--max_images',    type=int, default=3000,
                        help='Limit number of FFHQ images (e.g. 5000 for quick test)')
    parser.add_argument('--download_ffhq', action='store_true',
                        help='Download FFHQ thumbnails from Hugging Face')
    main(parser.parse_args())