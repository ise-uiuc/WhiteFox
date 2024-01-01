
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 3)
        self.bn = torch.nn.BatchNorm2d(3)
        self.relu = torch.nn.ReLU()
    def forward(self, x):
        b = self.relu(self.conv(self.bn(x)))
        return b
# Inputs to the model
x = torch.randn(1, 3, 4, 4)
