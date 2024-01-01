
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 1, 1)
        self.conv2 = torch.nn.Conv2d(1, 1, 1)
        self.bn = torch.nn.BatchNorm2d(1)
        # Conv3D is a subclass of Conv2D, so Fuser needs to work for both
        self.conv3 = torch.nn.Conv3d(1, 1, 1)
        self.conv4 = torch.nn.Conv3d(1, 1, 1)
        self.bn2 = torch.nn.BatchNorm3d(1)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn(x)
        # Conv3D
        y = self.conv3(x)
        y = self.conv4(y)
        y = self.bn2(y)
        return y
# Inputs to the model
x = torch.randn(1, 1, 1, 1, 1)
