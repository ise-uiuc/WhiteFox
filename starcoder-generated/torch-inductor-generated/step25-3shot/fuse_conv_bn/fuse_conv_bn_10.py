
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, kernel_size=2, stride=1)
        self.bn = torch.nn.BatchNorm2d(3)
        self.relu = torch.nn.ReLU(inplace=True)
    def forward(self, x3):
        v = self.relu(self.bn(self.conv(x3)))
        return v
# Inputs to the model
x3 = torch.randn(1,3,6,6)
