"""
GAN-based Image Inpainting Model
=================================
Generator: U-Net encoder-decoder with gated convolutions (inspired by DeepFill v2)
Discriminator: PatchGAN (classifies overlapping patches as real/fake)

Input to Generator:  masked_image (3ch) + mask (1ch) = 4 channels
Output:              inpainted image (3ch), values in [-1, 1]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
#  Building blocks
# ---------------------------------------------------------------------------

class GatedConv2d(nn.Module):
    """Gated convolution: learns a soft attention gate per spatial location."""

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1,
                 activation=nn.ELU(inplace=True), use_bn=True):
        super().__init__()
        self.conv_feature = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding)
        self.conv_gate = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_ch) if use_bn else nn.Identity()
        self.activation = activation

    def forward(self, x):
        feat = self.conv_feature(x)
        feat = self.bn(feat)
        if self.activation is not None:
            feat = self.activation(feat)
        gate = torch.sigmoid(self.conv_gate(x))
        return feat * gate


class GatedDeconv2d(nn.Module):
    """Upsampling + gated convolution."""

    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1, use_bn=True):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.gated_conv = GatedConv2d(in_ch, out_ch, kernel_size, stride=1,
                                      padding=padding, use_bn=use_bn)

    def forward(self, x):
        x = self.up(x)
        return self.gated_conv(x)


# ---------------------------------------------------------------------------
#  Generator  (U-Net with gated convolutions)
# ---------------------------------------------------------------------------

class Generator(nn.Module):
    """
    U-Net encoder-decoder.

    Input:  [masked_image, mask]  →  (B, 4, 256, 256)
    Output: completed image       →  (B, 3, 256, 256)  in [-1, 1]
    """

    def __init__(self, in_ch=4, base_ch=48, noise_dim=32):
        super().__init__()
        c = base_ch  # 48
        self.noise_dim = noise_dim

        # ---- Encoder ----
        self.enc1 = GatedConv2d(in_ch, c, 5, stride=1, padding=2)       # 256
        self.enc2 = GatedConv2d(c,   c*2, 3, stride=2, padding=1)       # 128
        self.enc3 = GatedConv2d(c*2, c*4, 3, stride=2, padding=1)       # 64
        self.enc4 = GatedConv2d(c*4, c*8, 3, stride=2, padding=1)       # 32
        self.enc5 = GatedConv2d(c*8, c*8, 3, stride=2, padding=1)       # 16

        # ---- Bottleneck ----
        # noise_dim channels are projected and added here for stochastic output
        self.noise_proj = nn.Sequential(
            nn.Conv2d(noise_dim, c*8, 1),
            nn.ELU(inplace=True),
        )
        self.bottleneck = nn.Sequential(
            GatedConv2d(c*8, c*8, 3, stride=1, padding=1, use_bn=True),
            GatedConv2d(c*8, c*8, 3, stride=1, padding=1, use_bn=True),
        )

        # ---- Decoder (with skip connections) ----
        # dec5 upsamples bottleneck 16→32, then cat with enc4(32) happens after
        self.dec5_up = GatedDeconv2d(c*8, c*8)        # 16 → 32
        self.dec5_fuse = GatedConv2d(c*8 + c*8, c*8, 3, stride=1, padding=1)

        self.dec4_up = GatedDeconv2d(c*8, c*4)        # 32 → 64
        self.dec4_fuse = GatedConv2d(c*4 + c*4, c*4, 3, stride=1, padding=1)

        self.dec3_up = GatedDeconv2d(c*4, c*2)        # 64 → 128
        self.dec3_fuse = GatedConv2d(c*2 + c*2, c*2, 3, stride=1, padding=1)

        self.dec2_up = GatedDeconv2d(c*2, c)          # 128 → 256
        self.dec2_fuse = GatedConv2d(c + c, c, 3, stride=1, padding=1)

        self.final = nn.Sequential(
            nn.Conv2d(c, 3, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def forward(self, masked_img, mask, z=None):
        """
        Args:
            masked_img: (B, 3, H, W) — image with masked region zeroed out
            mask:       (B, 1, H, W) — 1 = hole, 0 = valid
            z:          (B, noise_dim) — optional noise vector for stochastic output
                        If None, sampled randomly (stochastic); pass same z for deterministic.
        Returns:
            output:     (B, 3, H, W) — full inpainted image in [-1, 1]
        """
        B = masked_img.size(0)
        x_in = torch.cat([masked_img, mask], dim=1)   # (B, 4, H, W)

        e1 = self.enc1(x_in)   # (B, c, 256, 256)
        e2 = self.enc2(e1)     # (B, 2c, 128, 128)
        e3 = self.enc3(e2)     # (B, 4c, 64, 64)
        e4 = self.enc4(e3)     # (B, 8c, 32, 32)
        e5 = self.enc5(e4)     # (B, 8c, 16, 16)

        # inject noise for stochastic output
        if z is None:
            z = torch.randn(B, self.noise_dim, device=masked_img.device)
        # reshape z to spatial: (B, noise_dim, 1, 1) → broadcast to (B, noise_dim, 16, 16)
        z_spatial = z.view(B, self.noise_dim, 1, 1).expand(-1, -1, e5.size(2), e5.size(3))
        e5 = e5 + self.noise_proj(z_spatial)

        b = self.bottleneck(e5)  # (B, 8c, 16, 16)

        d5 = self.dec5_fuse(torch.cat([self.dec5_up(b),  e4], dim=1))  # 32
        d4 = self.dec4_fuse(torch.cat([self.dec4_up(d5), e3], dim=1))  # 64
        d3 = self.dec3_fuse(torch.cat([self.dec3_up(d4), e2], dim=1))  # 128
        d2 = self.dec2_fuse(torch.cat([self.dec2_up(d3), e1], dim=1))  # 256

        output = self.final(d2)                        # (B, 3, 256, 256)
        return output


# ---------------------------------------------------------------------------
#  Discriminator  (PatchGAN — 70×70 receptive field)
# ---------------------------------------------------------------------------

class Discriminator(nn.Module):
    """
    PatchGAN discriminator.
    Input: image (3ch)  →  outputs a spatial map of real/fake scores.
    Uses spectral normalization for training stability.
    """

    def __init__(self, in_ch=3, base_ch=64):
        super().__init__()
        c = base_ch

        def disc_block(in_c, out_c, stride=2, use_bn=True):
            layers = [nn.utils.spectral_norm(
                nn.Conv2d(in_c, out_c, 4, stride=stride, padding=1)
            )]
            if use_bn:
                layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        self.model = nn.Sequential(
            disc_block(in_ch, c,   stride=2, use_bn=False),  # 256→128
            disc_block(c,   c*2, stride=2),                   # 128→64
            disc_block(c*2, c*4, stride=2),                   # 64→32
            disc_block(c*4, c*8, stride=1),                   # 32→31
            nn.Conv2d(c*8, 1, kernel_size=4, padding=1),      # 31→30
        )

    def forward(self, x):
        return self.model(x)


# ---------------------------------------------------------------------------
#  Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    device = 'cpu'
    B = 2

    masked = torch.randn(B, 3, 256, 256, device=device)
    mask   = torch.zeros(B, 1, 256, 256, device=device)
    mask[:, :, 64:192, 64:192] = 1.0

    G = Generator().to(device)
    D = Discriminator().to(device)

    out = G(masked, mask)
    print(f'Generator output : {out.shape}')   # (2, 3, 256, 256)

    score = D(out)
    print(f'Discriminator out: {score.shape}')  # (2, 1, 30, 30)

    total_g = sum(p.numel() for p in G.parameters())
    total_d = sum(p.numel() for p in D.parameters())
    print(f'Generator params : {total_g / 1e6:.1f}M')
    print(f'Discriminator params: {total_d / 1e6:.1f}M')