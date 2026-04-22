import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.shared.dataset import CelebAHQDataset
from src.shared.mask import batch_masks
from src.shared.evaluate import evaluate_all
from src.reconstruction.model import UNetInpainting

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}", flush=True)

    dataset = CelebAHQDataset(args.data_dir, image_size=args.image_size)
    n_val   = int(len(dataset) * 0.1)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_set,   batch_size=args.batch_size, shuffle=False, num_workers=0)

    model     = UNetInpainting().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.L1Loss()

    os.makedirs('models', exist_ok=True)

    best_ssim = 0
    patience  = 0

    for epoch in range(args.epochs):
        model.train()
        for i, x in enumerate(train_loader):
            x    = x.to(device)
            B, C, H, W = x.shape
            mask = batch_masks(B, H, W, condition='medium').to(device)
            y    = x * (1 - mask)
            x_hat = model(y, mask)
            loss  = criterion(x_hat * mask, x * mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                print(f"Epoch {epoch+1} batch {i}/{len(train_loader)} loss: {loss.item():.4f}", flush=True)

        model.eval()
        val_metrics = []
        with torch.no_grad():
            for x in val_loader:
                x    = x.to(device)
                B, C, H, W = x.shape
                mask = batch_masks(B, H, W, condition='medium').to(device)
                y    = x * (1 - mask)
                x_hat = model(y, mask)
                val_metrics.append(evaluate_all(x, x_hat))

        avg_mse  = sum(m['mse']  for m in val_metrics) / len(val_metrics)
        avg_ssim = sum(m['ssim'] for m in val_metrics) / len(val_metrics)
        print(f"Epoch {epoch+1}/{args.epochs} done  Loss: {loss.item():.4f}  Val MSE: {avg_mse:.4f}  Val SSIM: {avg_ssim:.4f}", flush=True)


        torch.save(model.state_dict(), f"models/reconstruction_epoch{epoch+1}.pth")

        if avg_ssim > best_ssim:
            best_ssim = avg_ssim
            patience  = 0
            torch.save(model.state_dict(), "models/reconstruction_best.pth")
            print(f"  ✓ Best model saved (SSIM={best_ssim:.4f})", flush=True)
        else:
            patience += 1
            print(f"  No improvement ({patience}/{args.patience})", flush=True)
            if patience >= args.patience:
                print(f"Early stopping at epoch {epoch+1}", flush=True)
                break

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',   default='data/celeba-hq')
    parser.add_argument('--epochs',     type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--patience',   type=int, default=5)
    train(parser.parse_args())
