import torch
import torch.nn as nn


def _conv_block(in_c, out_c):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 4, stride=2, padding=1),
        nn.BatchNorm2d(out_c),
        nn.LeakyReLU(0.2, inplace=True),
    )


def _up_block(in_c, out_c):
    return nn.Sequential(
        nn.ConvTranspose2d(in_c, out_c, 4, stride=2, padding=1),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
    )


class Encoder(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()
        self.b1 = _conv_block(4, 64)      # 256 -> 128
        self.b2 = _conv_block(64, 128)    # 128 -> 64
        self.b3 = _conv_block(128, 256)   # 64 -> 32
        self.b4 = _conv_block(256, 512)   # 32 -> 16
        self.b5 = _conv_block(512, 512)   # 16 -> 8
        self.b6 = _conv_block(512, 512)   # 8 -> 4
        self.to_mu = nn.Conv2d(512, latent_dim, 1)
        self.to_logvar = nn.Conv2d(512, latent_dim, 1)

    def forward(self, y, mask):
        h0 = torch.cat([y, mask], dim=1)
        h1 = self.b1(h0)
        h2 = self.b2(h1)
        h3 = self.b3(h2)
        h4 = self.b4(h3)
        h5 = self.b5(h4)
        h6 = self.b6(h5)
        mu = self.to_mu(h6)
        logvar = self.to_logvar(h6)
        return mu, logvar, [h1, h2, h3, h4, h5]


class Decoder(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()
        self.from_z = nn.Sequential(
            nn.Conv2d(latent_dim, 512, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.up1 = _up_block(512, 512)              # 4 -> 8
        self.up2 = _up_block(512 + 512, 512)        # 8 -> 16 (+h5)
        self.up3 = _up_block(512 + 512, 256)        # 16 -> 32 (+h4)
        self.up4 = _up_block(256 + 256, 128)        # 32 -> 64 (+h3)
        self.up5 = _up_block(128 + 128, 64)         # 64 -> 128 (+h2)
        self.up6 = _up_block(64 + 64, 32)           # 128 -> 256 (+h1)
        self.out_conv = nn.Conv2d(32, 3, 3, padding=1)

    def forward(self, z, skips):
        h1, h2, h3, h4, h5 = skips
        h = self.from_z(z)
        h = self.up1(h)
        h = self.up2(torch.cat([h, h5], dim=1))
        h = self.up3(torch.cat([h, h4], dim=1))
        h = self.up4(torch.cat([h, h3], dim=1))
        h = self.up5(torch.cat([h, h2], dim=1))
        h = self.up6(torch.cat([h, h1], dim=1))
        return torch.tanh(self.out_conv(h))


class InpaintingVAE(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def forward(self, y, mask):
        mu, logvar, skips = self.encoder(y, mask)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decoder(z, skips)
        return x_hat, mu, logvar

    @torch.no_grad()
    def inpaint(self, y, mask, stochastic=True):
        mu, logvar, skips = self.encoder(y, mask)
        z = self.reparameterize(mu, logvar) if stochastic else mu
        raw = self.decoder(z, skips)
        return y * (1 - mask) + raw * mask


if __name__ == "__main__":
    model = InpaintingVAE()
    y = torch.randn(2, 3, 256, 256)
    m = torch.zeros(2, 1, 256, 256)
    m[:, :, 64:128, 64:128] = 1
    x_hat, mu, logvar = model(y, m)
    print(f"x_hat: {tuple(x_hat.shape)}, params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
