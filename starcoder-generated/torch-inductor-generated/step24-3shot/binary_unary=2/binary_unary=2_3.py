
import torchvision.transforms.functional as TF
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 2, stride=1, padding=3, dilation=2)
        self.transform = TF.vflip
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.mean(v1)
        v3 = torch.sigmoid(v2)
        v4 = self.transform(v3)
        return torch.nn.functional.conv2d(v4, weight = 1, bias= 10)
# Inputs to the model
x1 = torch.randn(1, 3, 128, 64)
