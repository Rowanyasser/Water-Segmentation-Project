import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
        )
    def forward(self, x): return self.net(x)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.1):
        super().__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch, dropout=dropout)
        )
    def forward(self, x): return self.mpconv(x)

class Up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True, dropout=0.3):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_ch, out_ch, dropout=dropout)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_ch, out_ch, dropout=dropout)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class TransformerBlock(nn.Module):
    def __init__(self, emb_size, num_heads=8, ff_hidden=1024, dropout=0.3):
        super().__init__()
        self.attn = nn.MultiheadAttention(emb_size, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(emb_size)
        self.ff = nn.Sequential(
            nn.Linear(emb_size, ff_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_hidden, emb_size),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(emb_size)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ff(x))
        return x

class TransformerBottleneck(nn.Module):
    def __init__(self, in_ch, emb_size=256, patch_size=4, depth=2, num_heads=8, dropout=0.3):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_ch, emb_size, kernel_size=patch_size, stride=patch_size)
        self.blocks = nn.ModuleList([
            TransformerBlock(emb_size, num_heads, emb_size*4, dropout) for _ in range(depth)
        ])
        self.deproj = nn.Linear(emb_size, in_ch)  # Initialize deproj to match saved weights

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)  # (B, emb_size, H/patch_size, W/patch_size)
        Hn, Wn = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)  # (B, N, emb_size)
        for blk in self.blocks:
            x = blk(x)
        x = self.deproj(x)  # (B, N, C)
        x = x.transpose(1, 2).contiguous().view(B, C, Hn, Wn)
        return F.interpolate(x, size=(H, W), mode="bilinear", align_corners=True)

class TransUNet(nn.Module):
    def __init__(self, in_ch=22, n_classes=1, base_c=32, bilinear=True, dropout=0.3):
        super().__init__()
        self.inc = DoubleConv(in_ch, base_c, dropout=dropout)
        self.down1 = Down(base_c, base_c*2, dropout=dropout)
        self.down2 = Down(base_c*2, base_c*4, dropout=dropout)
        self.down3 = Down(base_c*4, base_c*8, dropout=dropout)
        factor = 2 if bilinear else 1
        self.down4 = Down(base_c*8, base_c*16 // factor, dropout=dropout)
        self.trans = TransformerBottleneck(base_c*16 // factor, emb_size=256, patch_size=4, depth=2, num_heads=8, dropout=dropout)
        self.up1 = Up(base_c*16, base_c*8 // factor, bilinear, dropout=dropout)
        self.up2 = Up(base_c*8, base_c*4 // factor, bilinear, dropout=dropout)
        self.up3 = Up(base_c*4, base_c*2 // factor, bilinear, dropout=dropout)
        self.up4 = Up(base_c*2, base_c, bilinear, dropout=dropout)
        self.outc = nn.Conv2d(base_c, n_classes, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x5 = self.trans(x5)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)

if __name__ == "__main__":
    model = TransUNet(in_ch=22, n_classes=1)
    x = torch.randn(1, 22, 256, 256)
    y = model(x)
    print(f"Output shape: {y.shape}")