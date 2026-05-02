import torch.nn as nn
import torchvision.models as models


class SimCLR(nn.Module):
    def __init__(self, base_model=models.resnet50, out_dim=128):
        super(SimCLR, self).__init__()
        self.backbone = base_model(weights=models.ResNet50_Weights.DEFAULT)
        dim_mlp = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        self.projection = nn.Sequential(
            nn.Linear(dim_mlp, 512), nn.ReLU(), nn.Linear(512, out_dim)
        )

    def forward(self, x):
        h = self.backbone(x)
        z = self.projection(h)
        return h, z
