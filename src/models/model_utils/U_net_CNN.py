import torch
import torch.nn as nn
import torchvision.transforms as TF


class DoubleConv(nn.Module):
    """U-net double convolution block: (CNN => ReLU => BN) * 2"""
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_batch_norm=False,
                 ):
        super().__init__()
        block = []
        block.append(nn.Conv2d(in_channels, out_channels,
                               kernel_size=3, stride=1, padding=1))
        if use_batch_norm:
            block.append(nn.BatchNorm2d(out_channels))
        block.append(nn.ReLU(inplace=True))

        block.append(nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, stride=1, padding=1))
        if use_batch_norm:
            block.append(nn.BatchNorm2d(out_channels))
        block.append(nn.ReLU(inplace=True))

        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out


class Encoder(nn.Module):
    """U-net encoder"""
    def __init__(self, chs=(3, 64, 128, 256, 512, 1024)):
        super().__init__()
        self.enc_blocks = nn.ModuleList(
            [DoubleConv(chs[i], chs[i + 1]) for i in range(len(chs) - 1)])
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        ftrs = []
        for block in self.enc_blocks:
            x = block(x)
            ftrs.append(x)
            x = self.pool(x)
        return ftrs


class UpConv(nn.Module):
    """U-net Up-Conv layer. Can be real Up-Conv or bilinear up-sampling"""
    def __init__(self,
                 in_channels,
                 out_channels,
                 up_mode='bilinear',
                 ):
        super().__init__()
        if up_mode == 'upconv':
            self.up = nn.ConvTranspose2d(
                in_channels, out_channels,
                kernel_size=2, stride=2, padding=0)
        elif up_mode == 'bilinear':
            self.up = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2,
                            align_corners=False),
                nn.Conv2d(in_channels, out_channels, kernel_size=3,
                          stride=1, padding=1))
        else:
            raise ValueError("No such up_mode")

    def forward(self, x):
        return self.up(x)


class Decoder(nn.Module):
    """U-net decoder, made of up-convolutions and CNN blocks.
    The cropping is necessary when 0-padding, due to the loss of
    border pixels in every convolution"""
    def __init__(self, chs=(1024, 512, 256, 128, 64)):
        super().__init__()
        self.chs = chs
        self.upconvs = nn.ModuleList(
            [UpConv(chs[i], chs[i + 1]) for i in range(len(chs) - 1)])
        self.dec_blocks = nn.ModuleList(
            [DoubleConv(2*chs[i + 1], chs[i + 1]) for i in range(len(chs) - 1)])

    def center_crop(self, enc_ftrs, x):
        _, _, H, W = x.shape
        enc_ftrs = TF.CenterCrop([H, W])(enc_ftrs)
        return enc_ftrs

    def forward(self, x, encoder_features):
        for i in range(len(self.chs) - 1):
            x = self.upconvs[i](x)
            try:
                enc_ftrs = encoder_features[i]
                x = torch.cat([x, enc_ftrs], dim=1)
            except RuntimeError:
                enc_ftrs = self.center_crop(encoder_features[i], x)
                x = torch.cat([x, enc_ftrs], dim=1)
            x = self.dec_blocks[i](x)
        return x


class OutputLayer(nn.Module):
    """U-net output layer: (CNN 1x1 => Sigmoid)"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.out_layer = nn.Conv2d(in_channels, out_channels,
                                   kernel_size=1, stride=1, padding=0)
        # TODO: do not use sigmoid if you use BCEWithLogitsLoss
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.out_layer(x)


class UNet(nn.Module):
    """U-net architecture.
    retain_dim interpolate the output to have the same
    dimension of the input image"""
    def __init__(self,
                 enc_chs=(3, 64, 128, 256, 512, 1024),
                 dec_chs=(1024, 512, 256, 128, 64),
                 out_chs=1,
                 retain_dim=True,
                 ):
        super().__init__()
        self.encoder = Encoder(enc_chs)
        self.decoder = Decoder(dec_chs)
        self.head = OutputLayer(dec_chs[-1], out_chs)
        self.retain_dim = retain_dim

    def forward(self, x):
        _, _, H, W = x.shape
        enc_ftrs = self.encoder(x)
        out = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:])
        out = self.head(out)
        if self.retain_dim:
            out = nn.functional.interpolate(out, (H, W))
        return out
