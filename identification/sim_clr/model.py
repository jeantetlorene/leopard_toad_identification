import torch
import torch.nn as nn
import torchvision.models as models


class SimCLR(nn.Module):
    def __init__(self, base_model=models.resnet50, out_dim=128, pretrained_path=None):
        super(SimCLR, self).__init__()
        self.backbone = base_model(weights=None)

        if pretrained_path:
            print(f"Loading pretrained backbone from {pretrained_path}")
            checkpoint = torch.load(pretrained_path, map_location="cpu")
            state_dict = checkpoint.get("model", checkpoint)

            # Filter and rename keys for standard ResNet50
            # Faster R-CNN backbone body uses 'backbone.body.' prefix
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("backbone.body."):
                    new_key = k.replace("backbone.body.", "")
                    new_state_dict[new_key] = v

            self.backbone.load_state_dict(new_state_dict, strict=False)
        else:
            # Fallback to ImageNet weights if no custom path provided
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


class SimCLRBackbone(nn.Module):
    def __init__(self, base_model=models.resnet50):
        super(SimCLRBackbone, self).__init__()
        self.backbone = base_model(weights=None)
        self.backbone.fc = nn.Identity()

    def forward(self, x):
        return self.backbone(x)
