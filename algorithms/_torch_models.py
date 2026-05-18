"""
Wspólne definicje modeli PyTorch dla treningu i inferencji.
"""


def build_unet(torch, in_channels, num_classes, base_channels=32):
    class DoubleConv(torch.nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.block = torch.nn.Sequential(
                torch.nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
                torch.nn.BatchNorm2d(out_ch),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
                torch.nn.BatchNorm2d(out_ch),
                torch.nn.ReLU(inplace=True),
            )

        def forward(self, x):
            return self.block(x)

    class UNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            c = base_channels
            self.enc1 = DoubleConv(in_channels, c)
            self.enc2 = torch.nn.Sequential(torch.nn.MaxPool2d(2), DoubleConv(c, c * 2))
            self.enc3 = torch.nn.Sequential(torch.nn.MaxPool2d(2), DoubleConv(c * 2, c * 4))
            self.enc4 = torch.nn.Sequential(torch.nn.MaxPool2d(2), DoubleConv(c * 4, c * 8))
            self.bottleneck = torch.nn.Sequential(
                torch.nn.MaxPool2d(2),
                DoubleConv(c * 8, c * 16),
            )
            self.dec4 = DoubleConv(c * 16 + c * 8, c * 8)
            self.dec3 = DoubleConv(c * 8 + c * 4, c * 4)
            self.dec2 = DoubleConv(c * 4 + c * 2, c * 2)
            self.dec1 = DoubleConv(c * 2 + c, c)
            self.head = torch.nn.Conv2d(c, num_classes, kernel_size=1)

        def forward(self, x):
            s1 = self.enc1(x)
            s2 = self.enc2(s1)
            s3 = self.enc3(s2)
            s4 = self.enc4(s3)
            b = self.bottleneck(s4)
            d4 = torch.nn.functional.interpolate(
                b, size=s4.shape[2:], mode='bilinear', align_corners=False
            )
            d4 = self.dec4(torch.cat([d4, s4], dim=1))
            d3 = torch.nn.functional.interpolate(
                d4, size=s3.shape[2:], mode='bilinear', align_corners=False
            )
            d3 = self.dec3(torch.cat([d3, s3], dim=1))
            d2 = torch.nn.functional.interpolate(
                d3, size=s2.shape[2:], mode='bilinear', align_corners=False
            )
            d2 = self.dec2(torch.cat([d2, s2], dim=1))
            d1 = torch.nn.functional.interpolate(
                d2, size=s1.shape[2:], mode='bilinear', align_corners=False
            )
            d1 = self.dec1(torch.cat([d1, s1], dim=1))
            return self.head(d1)

    return UNet()


def build_model(
    torch,
    model_type,
    in_channels,
    num_classes,
    base_channels=32,
    encoder_name='resnet34',
):
    model_type = str(model_type).lower()
    if model_type == 'unet':
        return build_unet(
            torch,
            in_channels=in_channels,
            num_classes=num_classes,
            base_channels=base_channels,
        )

    try:
        import segmentation_models_pytorch as smp
    except ImportError as e:
        raise ImportError(
            'Brak biblioteki segmentation_models_pytorch. '
            'Zainstaluj: pip install segmentation-models-pytorch timm'
        ) from e

    arch_map = {
        'deeplabv3plus': smp.DeepLabV3Plus,
        'linknet': smp.Linknet,
        'fpn': smp.FPN,
        'pspnet': smp.PSPNet,
        'manet': smp.MAnet,
    }
    if model_type not in arch_map:
        raise ValueError(f'Nieobsługiwany typ modelu: {model_type}')

    model_cls = arch_map[model_type]
    return model_cls(
        encoder_name=encoder_name,
        encoder_weights=None,
        in_channels=in_channels,
        classes=num_classes,
    )
