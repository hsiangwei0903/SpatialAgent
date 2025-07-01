import torchvision.models as models
import torch.nn as nn
import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def build_dist_model(model_cfg):
    """
    Build a distance regressor model based on the provided configuration.
    
    Args:
        model_cfg (dict): Configuration dictionary containing model parameters.
    
    Returns:
        nn.Module: The constructed distance regressor model.
    """
    input_channels = model_cfg.get('input_channels', 5)
    backbone = model_cfg.get('backbone', 'resnet50')

    # load model from path
    if 'model_path' in model_cfg:
        model_path = model_cfg['model_path']
        if not model_path.endswith('.pth'):
            raise ValueError("Model path must end with .pth")
        model = ResNetDistanceRegressor(input_channels=input_channels, backbone=backbone, pretrained=False)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval()
        model.to(DEVICE)
    
    return model


class ResNetDistanceRegressor(nn.Module):
    def __init__(self, input_channels=5, backbone='resnet50', pretrained=False):
        super().__init__()
        self.resnet = getattr(models, backbone)(weights='IMAGENET1K_V1' if pretrained else None)

        # Replace first conv layer to accept custom number of input channels
        old_conv = self.resnet.conv1
        self.resnet.conv1 = nn.Conv2d(input_channels, old_conv.out_channels,
                                      kernel_size=old_conv.kernel_size,
                                      stride=old_conv.stride,
                                      padding=old_conv.padding,
                                      bias=old_conv.bias is not None)

        # Copy weights if input_channels == 3
        if pretrained and input_channels == 6:
            with torch.no_grad():
                self.resnet.conv1.weight[:, :3] = old_conv.weight
                self.resnet.conv1.weight[:, 3:] = old_conv.weight.clone()

        num_feats = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_feats, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.resnet(x).squeeze(1)  # Output shape: (B,)
