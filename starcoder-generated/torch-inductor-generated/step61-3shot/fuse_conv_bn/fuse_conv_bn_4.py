
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.ConvTranspose2d(16, 32, 3, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.conv2 = torch.nn.ConvTranspose3d(32, 16, 3)
        self.bn = torch.nn.BatchNorm3d(16)
        self.relu2 = torch.nn.ReLU(inplace=False)
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        return self.bn(x)
# Inputs to the model
x = torch.randn(1, 16, 16, 16, 16)
