
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 3)
        self.bn = torch.nn.BatchNorm2d(3)
        self.relu = torch.nn.ReLU()
        self.pool3d = torch.nn.AvgPool3d(2)
    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.bn(x)
        x = self.pool3d(x)
        return x
# Inputs to the model
x = torch.randn(1, 3, 4, 4)
