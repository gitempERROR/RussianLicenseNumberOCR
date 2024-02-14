import torch
import torch.nn as nn

# tuple - out_size, kernel_size, stride
architecture = (
    (32, 3, 1), (64, 3, 2),
    1,
    (128, 3, 2),
    2,
    (256, 3, 2),
    8,
    (512, 3, 2),
    8,
    (1024, 3, 2),
    4,
    (512, 1, 1), (1024, 3, 1),
    "S",
    (256, 1, 1),
    "U",
    (256, 1, 1), (512, 3, 1),
    "S",
    (128, 1, 1),
    "U",
    (128, 1, 1), (256, 3, 1),
    "S"
)


class CNNBlock(nn.Module):
    def __init__(self, in_size: int, out_size: int, use_bn_act=True, **kwargs):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size

        if use_bn_act:
            self.conv = nn.Sequential(
                nn.Conv2d(in_size, out_size, bias=False, **kwargs),
                nn.BatchNorm2d(out_size),
                nn.LeakyReLU(0.1)
            )
        else:
            self.conv = nn.Conv2d(in_size, out_size, bias=True, **kwargs)

    def forward(self, x: torch.tensor):
        return self.conv(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_size: int, num_repeats: int, use_residual=True):
        super().__init__()
        self.in_size = in_size
        self.use_residual = use_residual
        self.conv = nn.ModuleList()
        self.num_repeats = num_repeats
        for _ in range(self.num_repeats):
            repeat = nn.Sequential(
                CNNBlock(in_size, in_size // 2, kernel_size=1),
                CNNBlock(in_size // 2, in_size, kernel_size=3, padding = 1),
            )
            self.conv.append(repeat)

    def forward(self, x: torch.tensor):
        if self.use_residual:
            for module in self.conv:
                x += module(x)
        else:
            for module in self.conv:
                x = module(x)
        return x


class ScaledPrediction(nn.Module):
    def __init__(self, in_size: int):
        super().__init__()
        self.in_size = in_size
        self.conv = nn.Sequential(
            CNNBlock(in_size, in_size * 2, kernel_size=3, padding=1),
            CNNBlock(in_size * 2, 15, use_bn_act=False ,kernel_size=1)
        )

    def forward(self, x: torch.tensor):
        x = self.conv(x)
        x = x.reshape(x.shape[0], 5, 3, x.shape[2], x.shape[3])  # test without permute
        return x


class YoloV3(nn.Module):
    def __init__(self, in_size=3):
        super().__init__()
        self.in_size = in_size
        self.layers = self._create_conv_layers()

    def forward(self, x):
        predictions = []
        skip_connections = []
        for layer in self.layers:
            if isinstance(layer, ScaledPrediction):
                predictions.append(layer(x))
                continue

            x = layer(x)

            if isinstance(layer, ResidualBlock) and layer.num_repeats == 8:
                skip_connections.append(x)
                continue

            if isinstance(layer, nn.Upsample):
                torch.cat((x, skip_connections[-1]), dim=1)
                skip_connections.pop()

        return predictions


    def _create_conv_layers(self):
        in_size = self.in_size
        layers = nn.ModuleList()
        for layer in architecture:
            if isinstance(layer, tuple):
                out_size, kernel_size, stride = layer
                padding = kernel_size // 3
                layers.append(
                    CNNBlock(in_size, out_size, kernel_size=kernel_size, stride=stride, padding=padding)
                )
                in_size = out_size
                continue

            if isinstance(layer, int):
                layers.append(
                    ResidualBlock(in_size, layer)
                )
                continue

            if layer == 'S':
                layers += [
                    ResidualBlock(in_size, 1, use_residual=False),
                    CNNBlock(in_size, in_size//2, kernel_size=1),
                    ScaledPrediction(in_size//2)
                ]
                continue

            if layer == 'U':
                layers.append(
                    nn.Upsample(scale_factor=2)
                )
                in_size = in_size * 3

        return layers
