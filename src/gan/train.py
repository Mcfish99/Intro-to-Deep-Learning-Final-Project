"""
GAN Inpainting — Training & Evaluation Script
================================================
Usage:
    python -m src.GAN.train train  --data_dir data/celeba_hq/celeba_hq_256 --epochs 50
    python -m src.GAN.train eval   --data_dir data/celeba_hq/celeba_hq_256 --ckpt_path models/gan/ckpt_epoch050.pt

Loss = L_adv + λ_rec * L_rec + λ_perc * L_perc
  - L_adv:  LSGAN adversarial loss (stable training)
  - L_rec:  L1 reconstruction loss
  - L_perc: perceptual loss (VGG16 features)
"""

import os
import sys
import argparse
import random
import time
import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
from torch.utils.data import DataLoader, Subset

# ---- project imports (run from repo root) ----
from src.shared.dataset import CelebAHQDataset
from src.shared.mask import batch_masks, apply_mask
from src.shared.evaluate import evaluate_all
from src.GAN.model import Generator, Discriminator


# ---------------------------------------------------------------------------
#  Perceptual Loss (VGG16 feature matching)
# ---------------------------------------------------------------------------

class PerceptualLoss(nn.Module):
    """L1 distance in VGG16 feature space (relu1_2, relu2_2, relu3_3)."""

    def __init__(self, device='cpu'):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features.eval()
        for p in vgg.parameters():
            p.requires_grad = False

        self.slice1 = vgg[:4].to(device)    # relu1_2
        self.slice2 = vgg[4:9].to(device)   # relu2_2
        self.slice3 = vgg[9:16].to(device)  # relu3_3

        self.register_buffer('mean',
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device))
        self.register_buffer('std',
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device))

    def _normalize(self, x):
        x = (x + 1) / 2
        return (x - self.mean) / self.std

    def forward(self, pred, target):
        p = self._normalize(pred)
        t = self._normalize(target)
        loss = 0
        for s in [self.slice1, self.slice2, self.slice3]:
            p, t = s(p), s(t)
            loss += nn.functional.l1_loss(p, t)
        return loss


# ---------------------------------------------------------------------------
#  Training
# ---------------------------------------------------------------------------

def train(args):
    # ---- Device ----
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f'Using device: {device}')

    # ---- Data ----
    dataset = CelebAHQDataset(args.data_dir, image_size=args.image_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, pin_memory=True,
                        drop_last=True)
    print(f'Dataset: {len(dataset)} images, {len(loader)} batches/epoch')

    # ---- Models ----
    G = Generator(in_ch=4, base_ch=args.base_ch, noise_dim=args.noise_dim).to(device)
    D = Discriminator(in_ch=3, base_ch=64).to(device)

    total_g = sum(p.numel() for p in G.parameters())
    total_d = sum(p.numel() for p in D.parameters())
    print(f'Generator params: {total_g/1e6:.1f}M  Discriminator params: {total_d/1e6:.1f}M')

    # ---- Optimizers ----
    opt_G = torch.optim.Adam(G.parameters(), lr=args.lr_g, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=args.lr_d, betas=(0.5, 0.999))

    # ---- Losses ----
    perceptual_loss_fn = PerceptualLoss(device=device)
    l1_loss = nn.L1Loss()

    # ---- Dirs ----
    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(args.sample_dir, exist_ok=True)

    # mask conditions to sample from during training
    mask_conditions = ['small', 'medium', 'large']

    # ---- Training loop ----
    for epoch in range(1, args.epochs + 1):
        G.train(); D.train()
        epoch_d_loss = 0.0
        epoch_g_loss = 0.0
        t0 = time.time()

        for step, real_imgs in enumerate(loader):
            real_imgs = real_imgs.to(device)
            B, _, H, W = real_imgs.shape

            # randomly sample mask condition each step (train on all sizes)
            cond = random.choice(mask_conditions)
            masks = batch_masks(B, H, W, condition=cond).to(device)
            masked_imgs = apply_mask(real_imgs, masks)

            # ============================================================
            #  Train Discriminator
            # ============================================================
            opt_D.zero_grad()

            fake_imgs = G(masked_imgs, masks).detach()
            comp_imgs = real_imgs * (1 - masks) + fake_imgs * masks

            pred_real = D(real_imgs)
            pred_fake = D(comp_imgs)

            # LSGAN loss
            loss_D_real = torch.mean((pred_real - 1.0) ** 2)
            loss_D_fake = torch.mean(pred_fake ** 2)
            loss_D = 0.5 * (loss_D_real + loss_D_fake)

            loss_D.backward()
            opt_D.step()

            # ============================================================
            #  Train Generator
            # ============================================================
            opt_G.zero_grad()

            fake_imgs = G(masked_imgs, masks)
            comp_imgs = real_imgs * (1 - masks) + fake_imgs * masks

            pred_fake = D(comp_imgs)

            # adversarial loss (LSGAN)
            loss_adv = torch.mean((pred_fake - 1.0) ** 2)

            # reconstruction loss (L1 on whole image)
            loss_rec = l1_loss(fake_imgs, real_imgs)

            # perceptual loss
            loss_perc = perceptual_loss_fn(comp_imgs, real_imgs)

            loss_G = loss_adv + args.lambda_rec * loss_rec + args.lambda_perc * loss_perc

            loss_G.backward()
            opt_G.step()

            epoch_d_loss += loss_D.item()
            epoch_g_loss += loss_G.item()

            if (step + 1) % args.log_every == 0:
                print(f'  [{epoch}/{args.epochs}] step {step+1}/{len(loader)} '
                      f'D={loss_D.item():.4f}  G={loss_G.item():.4f}  '
                      f'(adv={loss_adv.item():.4f} rec={loss_rec.item():.4f} '
                      f'perc={loss_perc.item():.4f}) mask={cond}')

        # ---- Epoch summary ----
        elapsed = time.time() - t0
        avg_d = epoch_d_loss / len(loader)
        avg_g = epoch_g_loss / len(loader)
        print(f'Epoch {epoch}/{args.epochs}  D_loss={avg_d:.4f}  '
              f'G_loss={avg_g:.4f}  time={elapsed:.0f}s')

        # ---- Save samples (show stochastic diversity) ----
        if epoch % args.save_every == 0:
            save_samples(G, real_imgs, masks, masked_imgs,
                         epoch, args.sample_dir, device)

        # ---- Save checkpoint ----
        if epoch % args.save_every == 0:
            torch.save({
                'epoch': epoch,
                'G': G.state_dict(),
                'D': D.state_dict(),
                'opt_G': opt_G.state_dict(),
                'opt_D': opt_D.state_dict(),
            }, os.path.join(args.ckpt_dir, f'ckpt_epoch{epoch:03d}.pt'))
            print(f'  checkpoint saved')

    print('Training complete.')


@torch.no_grad()
def save_samples(G, real, masks, masked, epoch, out_dir, device):
    """Save a grid: [masked | sample1 | sample2 | ground truth]
    Two different noise samples show stochastic diversity."""
    G.eval()
    n = min(6, real.size(0))

    # two different completions from different noise vectors
    comp1 = real[:n] * (1 - masks[:n]) + G(masked[:n], masks[:n]) * masks[:n]
    comp2 = real[:n] * (1 - masks[:n]) + G(masked[:n], masks[:n]) * masks[:n]

    grid = torch.cat([masked[:n], comp1, comp2, real[:n]], dim=0)
    grid = (grid + 1) / 2  # [-1,1] → [0,1]
    torchvision.utils.save_image(grid, os.path.join(out_dir, f'epoch{epoch:03d}.png'),
                                 nrow=n, padding=2)
    G.train()


# ---------------------------------------------------------------------------
#  Evaluation (run after training)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(args):
    """Evaluate trained model on all three mask conditions (small/medium/large)."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # load model
    G = Generator(in_ch=4, base_ch=args.base_ch, noise_dim=args.noise_dim).to(device)
    ckpt = torch.load(args.ckpt_path, map_location=device)
    G.load_state_dict(ckpt['G'])
    G.eval()

    # data
    dataset = CelebAHQDataset(args.data_dir, image_size=args.image_size)
    eval_size = min(1000, len(dataset))
    eval_dataset = Subset(dataset, range(eval_size))
    loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers)

    print(f'Evaluating on {eval_size} images...\n')
    print(f'{"Condition":>10s}  {"MSE":>10s}  {"SSIM":>10s}')
    print('-' * 36)

    for cond in ['small', 'medium', 'large']:
        mse_total, ssim_total, count = 0.0, 0.0, 0

        for real_imgs in loader:
            real_imgs = real_imgs.to(device)
            B, _, H, W = real_imgs.shape

            masks = batch_masks(B, H, W, condition=cond).to(device)
            masked_imgs = apply_mask(real_imgs, masks)

            fake_imgs = G(masked_imgs, masks)
            comp_imgs = real_imgs * (1 - masks) + fake_imgs * masks

            metrics = evaluate_all(real_imgs, comp_imgs)
            mse_total += metrics['mse'] * B
            ssim_total += metrics['ssim'] * B
            count += B

        print(f'{cond:>10s}  {mse_total/count:10.6f}  {ssim_total/count:10.4f}')

    print('\nNote: Run FID separately using src/shared/evaluate.py compute_fid()')


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description='Train/Evaluate GAN inpainting')
    sub = p.add_subparsers(dest='mode', help='train or eval')

    # ---- train ----
    tr = sub.add_parser('train', help='Train the model')
    tr.add_argument('--data_dir',    type=str,   required=True)
    tr.add_argument('--image_size',  type=int,   default=256)
    tr.add_argument('--batch_size',  type=int,   default=8)
    tr.add_argument('--epochs',      type=int,   default=50)
    tr.add_argument('--lr_g',        type=float, default=1e-4)
    tr.add_argument('--lr_d',        type=float, default=1e-4)
    tr.add_argument('--base_ch',     type=int,   default=48)
    tr.add_argument('--noise_dim',   type=int,   default=32)
    tr.add_argument('--lambda_rec',  type=float, default=10.0,
                    help='weight for L1 reconstruction loss')
    tr.add_argument('--lambda_perc', type=float, default=1.0,
                    help='weight for perceptual loss')
    tr.add_argument('--num_workers', type=int,   default=4)
    tr.add_argument('--ckpt_dir',    type=str,   default='models/gan')
    tr.add_argument('--sample_dir',  type=str,   default='results/gan/samples')
    tr.add_argument('--log_every',   type=int,   default=50)
    tr.add_argument('--save_every',  type=int,   default=5)

    # ---- eval ----
    ev = sub.add_parser('eval', help='Evaluate trained model')
    ev.add_argument('--data_dir',    type=str,   required=True)
    ev.add_argument('--ckpt_path',   type=str,   required=True)
    ev.add_argument('--image_size',  type=int,   default=256)
    ev.add_argument('--batch_size',  type=int,   default=16)
    ev.add_argument('--base_ch',     type=int,   default=48)
    ev.add_argument('--noise_dim',   type=int,   default=32)
    ev.add_argument('--num_workers', type=int,   default=4)

    args = p.parse_args()

    if args.mode is None:
        p.print_help()
        sys.exit(1)

    return args


if __name__ == '__main__':
    args = parse_args()
    if args.mode == 'train':
        train(args)
    elif args.mode == 'eval':
        evaluate(args)