
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = torch.nn.Conv2d(4, 4, 3)
        self.bn0 = torch.nn.BatchNorm2d(4)
        self.pool0 = torch.nn.AvgPool2d(5)
        self.conv1 = torch.nn.Conv3d(4, 4, 3)
        self.bn1 = torch.nn.BatchNorm3d(4)
        self.pool1 = torch.nn.AvgPool3d(5)
    def forward(self, x2):
        return self.conv1(self.bn1(self.pool1(self.conv0(self.pool0(self.bn0(x2))))))
# Inputs to the model
x2 = torch.randn(1, 4, 4, 4)
