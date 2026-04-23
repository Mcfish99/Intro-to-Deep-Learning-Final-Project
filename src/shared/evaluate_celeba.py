import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import argparse
import sys
import csv
import torch
from torch.utils.data import DataLoader, random_split
import torchvision.utils as vutils
import lpips as lpips_lib

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.shared.dataset  import CelebAHQDataset
from src.shared.mask     import batch_masks
from src.shared.evaluate import compute_mse, compute_ssim, compute_lpips, compute_fid

from src.reconstruction.model import UNetInpainting as ReconModel
# from src.gan.model  import GANGenerator  as GANModel
# from src.cvae.model import CVAEGenerator as CVAEModel

MASK_CONDITIONS = ['small', 'medium', 'large']


def save_images(tensor, output_dir, prefix, start_idx):
    os.makedirs(output_dir, exist_ok=True)
    imgs = (tensor.clamp(-1, 1) + 1) / 2
    for i, img in enumerate(imgs):
        path = os.path.join(output_dir, f"{prefix}_{start_idx + i:05d}.png")
        vutils.save_image(img, path)


def evaluate_model_on_condition(model, loader, device, lpips_fn,
                                fake_dir, real_dir, condition):
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

            save_images(x_hat, fake_dir, 'fake', img_idx)
            save_images(x,     real_dir, 'real', img_idx)
            img_idx += B

    return (
        sum(mse_scores)   / len(mse_scores),
        sum(ssim_scores)  / len(ssim_scores),
        sum(lpips_scores) / len(lpips_scores),
    )


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

    # 1. Load CelebA-HQ, use 10% as test set (same split as training)
    print("\nLoading CelebA-HQ dataset...")
    full_dataset = CelebAHQDataset(args.data_dir, image_size=args.image_size)
    n_val   = int(len(full_dataset) * 0.1)
    n_train = len(full_dataset) - n_val
    _, test_set = random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    loader = DataLoader(test_set, batch_size=args.batch_size,
                        shuffle=False, num_workers=0)
    print(f"  Test set: {len(test_set)} images")

    # 2. LPIPS (shared across all models)
    lpips_fn = lpips_lib.LPIPS(net='vgg').to(device)

    # 3. Models — each saves to its own results folder
    models_cfg = [
        ('Reconstruction', ReconModel, args.recon_ckpt, 'results/reconstruction'),
        # ('GAN',           GANModel,   args.gan_ckpt,  'results/GAN'),
        # ('CVAE',          CVAEModel,  args.cvae_ckpt, 'results/CVAE'),
    ]

    for model_name, model_class, ckpt_path, output_dir in models_cfg:
        if not os.path.exists(ckpt_path):
            print(f"\nWARNING: checkpoint not found at '{ckpt_path}', skipping.")
            continue

        print(f"\n{'='*50}")
        print(f"Evaluating: {model_name}")
        os.makedirs(output_dir, exist_ok=True)
        model = load_model(model_class, ckpt_path, device)
        results = []

        for condition in MASK_CONDITIONS:
            print(f"\n  Mask condition: {condition}")
            fake_dir = os.path.join(output_dir, f'fake_{condition}')
            real_dir = os.path.join(output_dir, f'real_{condition}')

            avg_mse, avg_ssim, avg_lpips = evaluate_model_on_condition(
                model, loader, device, lpips_fn,
                fake_dir=fake_dir,
                real_dir=real_dir,
                condition=condition,
            )

            print(f"  Computing FID...")
            fid = compute_fid(real_dir, fake_dir, device=str(device))

            print(f"  MSE:   {avg_mse:.4f}")
            print(f"  SSIM:  {avg_ssim:.4f}")
            print(f"  LPIPS: {avg_lpips:.4f}")
            print(f"  FID:   {fid:.2f}")

            results.append({
                'Regime': condition.capitalize(),
                'Model':  model_name,
                'MSE':    round(avg_mse,   4),
                'SSIM':   round(avg_ssim,  4),
                'LPIPS':  round(avg_lpips, 4),
                'FID':    round(fid,       2),
            })

        # Save CSV into this model's own folder
        csv_path = os.path.join(output_dir, 'celeba_results.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['Regime', 'Model', 'MSE', 'SSIM', 'LPIPS', 'FID'])
            writer.writeheader()
            writer.writerows(results)

        print(f"\n  Results saved to: {csv_path}")
        print(f"\n  --- {model_name} Summary ---")
        for r in results:
            print(f"  {r['Regime']:8s}  MSE={r['MSE']}  SSIM={r['SSIM']}  "
                  f"LPIPS={r['LPIPS']}  FID={r['FID']}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CelebA-HQ evaluation across mask regimes')
    parser.add_argument('--data_dir',   default='data/celeba_hq_256')
    parser.add_argument('--recon_ckpt', default='models/reconstruction_best.pth')
    parser.add_argument('--gan_ckpt',   default='models/gan_best.pth')
    parser.add_argument('--cvae_ckpt',  default='models/cvae_best.pth')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--image_size', type=int, default=256)
    main(parser.parse_args())