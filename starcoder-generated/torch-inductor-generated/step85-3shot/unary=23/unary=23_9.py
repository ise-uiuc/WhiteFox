
import torchvision.models as models
import torch

class Model(torch.nn.Module)
    def __init__(self):
        super().__init__()
        self.resnet50 = models.resnet50(pretrained=True)
        for param in self.resnet50.parameters():
            param.requires_grad = False
    def forward(self, x1):
        v1 = self.resnet50(x1)
        return v2
        