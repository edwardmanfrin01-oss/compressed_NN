"""ResNet models for single-channel maps and three-channel image experiments.

conv1 is always randomly initialized, including for RGB, matching the existing
experiments. Default: train conv1/fc only and freeze BatchNorm statistics.
The state_dict keys remain compatible with the notebook models.
"""
import torch
import torchvision.models as models
import torch.nn as nn

class CustomResNet50(models.ResNet):
    def __init__(self, num_classes=10, pretrained=True, *, in_channels=1,
                 freeze_backbone=True, freeze_batchnorm=True):
        super().__init__(models.resnet.Bottleneck, [3, 4, 6, 3], num_classes=num_classes)
        self.freeze_batchnorm = freeze_batchnorm
        self.conv1 = nn.Conv2d(in_channels, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
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


class CustomResNet50HalfLayer4(CustomResNet50):
    """ResNet50 with 256 internal channels in each layer4 bottleneck."""

    def __init__(self, num_classes=10, *, in_channels=1,
                 freeze_backbone=False, freeze_batchnorm=False):
        super().__init__(num_classes, pretrained=False, in_channels=in_channels,
                         freeze_backbone=freeze_backbone, freeze_batchnorm=freeze_batchnorm)
        for index, block in enumerate(self.layer4):
            # Keep the residual projection, stride, and 2048-channel output.
            # In TorchVision, base_width=32 halves the internal width.
            narrow_block = models.resnet.Bottleneck(
                block.conv1.in_channels, block.conv3.out_channels // block.expansion,
                stride=block.stride, downsample=block.downsample, base_width=32,
            )
            for conv in (narrow_block.conv1, narrow_block.conv2, narrow_block.conv3):
                nn.init.kaiming_normal_(conv.weight, mode="fan_out", nonlinearity="relu")
            self.layer4[index] = narrow_block.requires_grad_(not freeze_backbone)

    def load_full_width_state_dict(self, state_dict):
        """Initialize from a trained CustomResNet50 with matching input and classes."""
        narrowed_state = dict(state_dict)
        for index, block in enumerate(self.layer4):
            prefix = f"layer4.{index}."
            first_weight = state_dict[prefix + "conv1.weight"]
            second_weight = state_dict[prefix + "conv2.weight"]
            third_weight = state_dict[prefix + "conv3.weight"]
            width = block.conv1.out_channels
            if (first_weight.shape[0] != 2 * width
                    or second_weight.shape[:2] != (2 * width, 2 * width)
                    or third_weight.shape[1] != 2 * width):
                raise ValueError("Expected a full-width ResNet50 layer4 checkpoint.")

            # Keep filters with the largest L1 norms. Score conv2 after
            # restricting its inputs to the conv1 filters that survived.
            first_channels = first_weight.abs().sum(dim=(1, 2, 3)).topk(width).indices.sort().values
            second_inputs = second_weight.index_select(1, first_channels)
            second_channels = second_inputs.abs().sum(dim=(1, 2, 3)).topk(width).indices.sort().values

            narrowed_state[prefix + "conv1.weight"] = first_weight.index_select(0, first_channels)
            narrowed_state[prefix + "conv2.weight"] = second_inputs.index_select(0, second_channels)
            narrowed_state[prefix + "conv3.weight"] = third_weight.index_select(1, second_channels)

            # BatchNorm affine parameters and running statistics follow the
            # selected channels. Scalar batch counters are copied unchanged.
            for name, channels in (("bn1", first_channels), ("bn2", second_channels)):
                for field in ("weight", "bias", "running_mean", "running_var"):
                    key = f"{prefix}{name}.{field}"
                    narrowed_state[key] = state_dict[key].index_select(0, channels)

        return self.load_state_dict(narrowed_state, strict=True)


# Model: One Channel ResNet18
class CustomResNet18(models.ResNet):
    def __init__(self, num_classes, pretrained=True, *, in_channels=1, freeze_backbone=True, freeze_batchnorm=True):
        super().__init__(models.resnet.BasicBlock, [2,2,2,2], num_classes=num_classes) # recall the BasicBlocks, used in the standard implementation of ResNet18
        self.freeze_batchnorm=freeze_batchnorm
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)

        # define and upload the weights before assigning them to the class
        if pretrained:
            weights = models.ResNet18_Weights.IMAGENET1K_V1
            state = weights.get_state_dict(progress=True)
            state = {k: v for k,v in state.items()
                     if not (k.startswith("conv1.") or k.startswith("fc."))}
            self.load_state_dict(state, strict=False) 
        nn.init.kaiming_normal_(self.conv1.weight, mode="fan_out", nonlinearity="relu")

        # if "freeze_backbone = True", only the first and the last layer will be trained
        for parameter in self.parameters():
            parameter.requires_grad = not freeze_backbone
        for layer in (self.conv1, self.fc):
            for parameter in layer.parameters():
                parameter.requires_grad = True

    def train(self, mode=True):
        super().train(mode) # recall the "train" method from the parent class (model.ResNet)

        # set in evaluation mode all the BatchNorm layers, to avoid their training
        for module in self.modules():
            if self.freeze_batchnorm and isinstance(module, nn.BatchNorm2d):
                module.eval()
        return self

    # Inherit ResNet.forward from the parent class: it already supports the configured conv1

