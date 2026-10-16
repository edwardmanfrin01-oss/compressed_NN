"""ResNet50 shared by single-channel maps and 3-channels (RGB or YCbCr) experiments.

conv1 is always randomly initialized, including for RGB, matching the existing
experiments. Default: train conv1/fc only and freeze BatchNorm statistics.
The state_dict keys remain compatible with the notebook models.
"""
import torchvision.models as models
import torch.nn as nn

class CustomResNet50(models.ResNet):
    def __init__(self, num_classes=10, pretrained=True, *, in_channels=1,
                 freeze_backbone=True, freeze_batchnorm=True):
        super().__init__(models.resnet.Bottleneck, [3, 4, 6, 3], num_classes=num_classes)
        self.freeze_batchnorm = freeze_batchnorm
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if pretrained:
            weights = models.ResNet50_Weights.IMAGENET1K_V1
            state = weights.get_state_dict(progress=True)
            state = {k: v for k, v in state.items()
                     if not (k.startswith("conv1.") or k.startswith("fc."))}
            self.load_state_dict(state, strict=False)
        nn.init.kaiming_normal_(self.conv1.weight, mode="fan_out", nonlinearity="relu")
        # Preserve the chosen experiment: train only the new conv1 and fc.
        for parameter in self.parameters():
            parameter.requires_grad = not freeze_backbone
        for layer in (self.conv1, self.fc):
            for parameter in layer.parameters():
                parameter.requires_grad = True

    def train(self, mode=True):
        super().train(mode)
        # requires_grad=False does not freeze BatchNorm running statistics.
        # Keep the frozen backbone statistics fixed as well.
        for module in self.modules():
            if self.freeze_batchnorm and isinstance(module, nn.BatchNorm2d):
                module.eval()
        return self
    # Inherit ResNet.forward: it already supports the configured conv1.
