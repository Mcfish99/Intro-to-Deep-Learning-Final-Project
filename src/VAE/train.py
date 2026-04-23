import argparse
import random
import sys
import time
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.utils import save_image


_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parents[2]))    
sys.path.insert(0, str(_THIS.parent))          

from src.shared.dataset import CelebAHQDataset  
from model import InpaintingVAE                 



def sample_mask(H, W, size_range=(0.1, 0.6)):
    lo, hi = size_range
    h = random.randint(int(lo * H), int(hi * H))
    w = random.randint(int(lo * W), int(hi * W))
    cx = random.randint(w // 2, W - w // 2)
    cy = random.randint(h // 2, H - h // 2)
    mask = torch.zeros(1, H, W)
    mask[:, cy - h//2 : cy + h//2, cx - w//2 : cx + w//2] = 1
    return mask


def sample_mask_batch(B, H, W, device="cpu"):
    return torch.stack([sample_mask(H, W) for _ in range(B)]).to(device)

class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        from torchvision.models import vgg16, VGG16_Weights
        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features.eval()
        for p in vgg.parameters():
            p.requires_grad = False
        self.vgg = vgg[:23]                  
        self.layers = {3, 8, 15, 22}      
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _prep(self, x):
        return ((x + 1) / 2 - self.mean) / self.std

    def forward(self, x_hat, x):
        hh, ht = self._prep(x_hat), self._prep(x)
        loss = 0.0
        for i, layer in enumerate(self.vgg):
            hh, ht = layer(hh), layer(ht)
            if i in self.layers:
                loss = loss + F.l1_loss(hh, ht)
        return loss


def vae_loss(x, x_hat, mu, logvar, mask, beta_kl=0.01):
    diff = torch.abs(x_hat - x)
    l_hole = (diff * mask).sum() / (mask.sum() * 3 + 1e-8)
    l_valid = (diff * (1 - mask)).sum() / ((1 - mask).sum() * 3 + 1e-8)
    l_recon = 6.0 * l_hole + 1.0 * l_valid

    l_kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=[1, 2, 3]).mean()
    total = l_recon + beta_kl * l_kl
    return total, {"loss": total.item(), "hole": l_hole.item(),
                   "valid": l_valid.item(), "kl": l_kl.item()}



def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train(args):
    device = pick_device()
    print(f"[train] device: {device}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "samples").mkdir(exist_ok=True)

    dataset = CelebAHQDataset(args.data_dir, image_size=args.image_size)
    print(f"[train] dataset size: {len(dataset)}")
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    model = InpaintingVAE(latent_dim=args.latent_dim).to(device)
    print(f"[train] params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    optimizer = Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.999))

    perceptual = None
    if args.lambda_perc > 0 and not args.overfit:
        print("[train] using VGG perceptual loss")
        perceptual = VGGPerceptualLoss().to(device).eval()

    model.train()
    step = 0
    t0 = time.time()
    first_loss, last_loss = None, None

    for epoch in range(args.epochs):
        for x in loader:
            x = x.to(device, non_blocking=True)
            B, _, H, W = x.shape
            mask = sample_mask_batch(B, H, W, device=device)
            y = x * (1 - mask)

            x_hat, mu, logvar = model(y, mask)
            loss, logs = vae_loss(x, x_hat, mu, logvar, mask, beta_kl=args.beta_kl)

            if perceptual is not None:
                x_hat_comp = y * (1 - mask) + x_hat * mask
                l_perc = perceptual(x_hat_comp, x)
                loss = loss + args.lambda_perc * l_perc
                logs["perc"] = l_perc.item()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            if first_loss is None:
                first_loss = logs["loss"]
            last_loss = logs["loss"]
            step += 1

            if step % args.log_every == 0:
                msg = (f"[ep {epoch:3d} | step {step:6d} | {time.time()-t0:6.1f}s] "
                       f"loss={logs['loss']:.4f} hole={logs['hole']:.4f} "
                       f"kl={logs['kl']:.2f}")
                if "perc" in logs:
                    msg += f" perc={logs['perc']:.4f}"
                print(msg, flush=True)

            if args.overfit and step >= args.overfit_steps:
                print(f"\nFirst loss: {first_loss:.4f}")
                print(f"Last loss:  {last_loss:.4f}")
                print("OK: loss decreased." if last_loss < first_loss * 0.5
                      else "WARNING: loss did not decrease much.")
                return


        model.eval()
        with torch.no_grad():
            x_s = next(iter(loader)).to(device)[:8]
            mask_s = sample_mask_batch(x_s.size(0), x_s.size(2), x_s.size(3), device=device)
            y_s = x_s * (1 - mask_s)
            xh_s = model.inpaint(y_s, mask_s)
            grid = torch.cat([x_s, y_s, xh_s], dim=0)
            save_image((grid + 1) / 2, out_dir / "samples" / f"epoch_{epoch:03d}.png",
                       nrow=x_s.size(0))
        model.train()

        torch.save({"epoch": epoch, "model_state": model.state_dict()},
                   out_dir / f"vae_epoch_{epoch:03d}.pt")
        torch.save(model.state_dict(), out_dir / "vae_latest.pt")
        print(f"[train] saved checkpoint for epoch {epoch}")


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=str, required=True)
    p.add_argument("--output-dir", type=str, default="checkpoints/vae")
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--latent-dim", type=int, default=256)
    p.add_argument("--beta-kl", type=float, default=0.01)
    p.add_argument("--lambda-perc", type=float, default=0.1)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--overfit", action="store_true",
                   help="Quick sanity check: train a fixed number of steps")
    p.add_argument("--overfit-steps", type=int, default=200)
    return p.parse_args()


if __name__ == "__main__":
    train(get_args())
