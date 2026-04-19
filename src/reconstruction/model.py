import torch
import torch.nn as nn
import torch.nn.functional as F

class PartialConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.mask_conv = nn.Conv2d(1, 1, kernel_size, stride, padding, bias=False)
        nn.init.constant_(self.mask_conv.weight, 1.0)
        for p in self.mask_conv.parameters():
            p.requires_grad = False

    def forward(self, x, mask):
        valid = 1 - mask
        with torch.no_grad():
            valid_out = (self.mask_conv(valid) > 0).float()
        sum_valid = self.mask_conv(valid).clamp(min=1e-8)
        x_out = self.conv(x * valid) / sum_valid
        x_out = self.bn(x_out)
        mask_out = 1 - valid_out
        return x_out, mask_out


class UNetInpainting(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder（4次下采样，256->128->64->32->16）
        self.enc1 = PartialConv2d(3,   64,  7, stride=1, padding=3)
        self.enc2 = PartialConv2d(64,  128, 5, stride=2, padding=2)
        self.enc3 = PartialConv2d(128, 256, 5, stride=2, padding=2)
        self.enc4 = PartialConv2d(256, 512, 3, stride=2, padding=1)

        # Decoder
        self.dec4 = nn.Sequential(nn.Conv2d(512+256, 256, 3, padding=1), nn.ReLU())
        self.dec3 = nn.Sequential(nn.Conv2d(256+128, 128, 3, padding=1), nn.ReLU())
        self.dec2 = nn.Sequential(nn.Conv2d(128+64,  64,  3, padding=1), nn.ReLU())
        self.dec1 = nn.Sequential(nn.Conv2d(64+3,    3,   3, padding=1), nn.Tanh())

    def forward(self, y, mask):
        e1, m1 = self.enc1(y,  mask);  e1 = F.relu(e1)
        e2, m2 = self.enc2(e1, m1);    e2 = F.relu(e2)
        e3, m3 = self.enc3(e2, m2);    e3 = F.relu(e3)
        e4, _  = self.enc4(e3, m3);    e4 = F.relu(e4)

        d = F.interpolate(e4, size=e3.shape[2:], mode='nearest')
        d = self.dec4(torch.cat([d, e3], dim=1))

        d = F.interpolate(d, size=e2.shape[2:], mode='nearest')
        d = self.dec3(torch.cat([d, e2], dim=1))

        d = F.interpolate(d, size=e1.shape[2:], mode='nearest')
        d = self.dec2(torch.cat([d, e1], dim=1))

        d = F.interpolate(d, size=y.shape[2:], mode='nearest')
        d = self.dec1(torch.cat([d, y], dim=1))

        return d
