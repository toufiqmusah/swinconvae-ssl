import torch
from torch import nn

from monai.networks.nets import SwinUNETR
from nnunet_mednext import MedNeXtBlock, MedNeXtDownBlock, MedNeXtUpBlock

class SwinVITBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SwinVITBlock, self).__init__()
        swin = SwinUNETR(in_channels=in_channels, out_channels=out_channels)
        self.svit = swin.swinViT

    def forward(self, x):
        return self.svit(x)

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, exp_r=4, kernel_size=3,
                 do_res=True, norm_type='group'):
        super(EncoderBlock, self).__init__()

        self.mednext_block = MedNeXtBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            exp_r=exp_r,
            kernel_size=kernel_size,
            do_res=do_res and (in_channels == out_channels),
            norm_type=norm_type,
            n_groups=None
        )
        self.down_block = MedNeXtDownBlock(
            in_channels=out_channels,
            out_channels=out_channels,
            exp_r=exp_r,
            kernel_size=kernel_size,
            do_res=do_res,
            norm_type=norm_type
        )

    def forward(self, x, swin_features=None):
        x = self.mednext_block(x)

        if swin_features is not None:
            if x.shape[2:] != swin_features.shape[2:]:
                swin_features = torch.nn.functional.interpolate(
                    swin_features,
                    size=x.shape[2:],
                    mode='trilinear',
                    align_corners=False
                )
            if x.shape[1] != swin_features.shape[1]:
                if not hasattr(self, 'channel_proj'):
                    self.channel_proj = nn.Conv3d(swin_features.shape[1], x.shape[1], 1).to(x.device)
                swin_features = self.channel_proj(swin_features)
            x = x + swin_features

        skip_features = x.clone()
        x = self.down_block(x)
        return x, skip_features

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels=None,
                 exp_r=4, kernel_size=3, norm_type='group'):
        super(DecoderBlock, self).__init__()
        self.skip_channels = skip_channels

        # Main upsampling block
        self.up_block = MedNeXtUpBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            exp_r=exp_r,
            kernel_size=kernel_size,
            do_res=False,
            norm_type=norm_type
        )

        # Skip feature processing
        if skip_channels is not None:
            self.skip_conv = nn.Sequential(
                nn.Conv3d(skip_channels, out_channels, kernel_size=1),
                nn.GroupNorm(8, out_channels),
                nn.ReLU(inplace=True)
            )

            # Changed to regular Conv block instead of MedNeXtBlock
            self.combine_conv = nn.Sequential(
                nn.Conv3d(out_channels * 2, out_channels, kernel_size=3, padding=1),
                nn.GroupNorm(8, out_channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.skip_conv = None
            self.combine_conv = None

    def forward(self, x, skip_features=None, use_skip=True):
        # Upsample first
        x = self.up_block(x)

        if use_skip and skip_features is not None and self.skip_channels is not None:

            # Process skip features
            skip = self.skip_conv(skip_features)

            if x.shape[2:] != skip.shape[2:]:
                skip = torch.nn.functional.interpolate(
                    skip,
                    size=x.shape[2:],
                    mode='trilinear',
                    align_corners=False
                )

            x = torch.cat([x, skip], dim=1)
            x = self.combine_conv(x)

        return x

class SwinConvAE(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, use_skip_connections=True):
        super(SwinConvAE, self).__init__()
        self.use_skip_connections = use_skip_connections
        self.swin_vit = SwinVITBlock(in_channels, out_channels)
        self.input_conv = nn.Conv3d(in_channels, 64, kernel_size=3, padding=1)

        # Encoder
        self.encoder1 = EncoderBlock(64, 64)
        self.encoder2 = EncoderBlock(64, 128)
        self.encoder3 = EncoderBlock(128, 256)
        self.encoder4 = EncoderBlock(256, 512)

        # Bottleneck
        self.bottleneck = nn.Sequential(MedNeXtBlock(512, 512, exp_r=4, kernel_size=3, do_res=True),
                                        MedNeXtBlock(512, 512, exp_r=4, kernel_size=3, do_res=True))

        self.bottleneck_proj = None

        # Decoder
        skip_channels = [64, 128, 256, 512] if use_skip_connections else [None]*4
        self.decoder4 = DecoderBlock(512, 256, skip_channels[3])
        self.decoder3 = DecoderBlock(256, 128, skip_channels[2])
        self.decoder2 = DecoderBlock(128, 64, skip_channels[1])
        self.decoder1 = DecoderBlock(64, 64, skip_channels[0])

        self.output_conv = nn.Conv3d(64, out_channels, kernel_size=1)

    def set_skip_connections(self, use_skip):
        self.use_skip_connections = use_skip

    def forward(self, x):
        swin_features = self.swin_vit(x)
        x = self.input_conv(x)

        skip_connections = []

        # Encoder path
        x, skip1 = self.encoder1(x, swin_features[0] if len(swin_features) > 0 else None)
        skip_connections.append(skip1)

        x, skip2 = self.encoder2(x, swin_features[1] if len(swin_features) > 1 else None)
        skip_connections.append(skip2)

        x, skip3 = self.encoder3(x, swin_features[2] if len(swin_features) > 2 else None)
        skip_connections.append(skip3)

        x, skip4 = self.encoder4(x, swin_features[3] if len(swin_features) > 3 else None)
        skip_connections.append(skip4)

        # Bottleneck
        if len(swin_features) > 4:
            bottleneck_swin = swin_features[4]
            if x.shape[2:] != bottleneck_swin.shape[2:]:
                bottleneck_swin = torch.nn.functional.interpolate(
                    bottleneck_swin, size=x.shape[2:], mode='trilinear', align_corners=False
                )
            if x.shape[1] != bottleneck_swin.shape[1]:
                if self.bottleneck_proj is None:
                    self.bottleneck_proj = nn.Conv3d(bottleneck_swin.shape[1], x.shape[1], 1).to(x.device)
                bottleneck_swin = self.bottleneck_proj(bottleneck_swin)
            x = x + bottleneck_swin

        x = self.bottleneck(x)

        # Reverse skip connections for decoder
        skip_connections = skip_connections[::-1]

        # Decoder path
        x = self.decoder4(x, skip_connections[0], self.use_skip_connections)
        x = self.decoder3(x, skip_connections[1], self.use_skip_connections)
        x = self.decoder2(x, skip_connections[2], self.use_skip_connections)
        x = self.decoder1(x, skip_connections[3], self.use_skip_connections)

        return self.output_conv(x)