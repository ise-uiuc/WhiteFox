
import torchvision.models as models
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        resnet = models.resnet18()
    def forward(self, x1, other=None):
        v1 = self.conv(x1)
        if other == None:
            other = torch.randn(v1.shape)
        v2 = v1 + other
        v3 = v2.mean()
        return v3, v2
# Inputs to the model
x1 = torch.randn(1, 3, 5, 5)
