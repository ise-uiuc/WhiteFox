
import torchvision
class Model (torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = torch.nn.Conv2d(3, 3, 3, 1, 1)
        self.bn = torch.nn.BatchNorm2d(3)
        self.conv2 = torchvision.models.alexnet.features[0]
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.conv2(x)
        return x
# Inputs to the model
x = torch.randn(1,3,256,256)
