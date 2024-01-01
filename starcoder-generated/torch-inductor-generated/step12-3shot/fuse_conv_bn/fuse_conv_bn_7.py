
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 3)
        self.bn = torch.nn.BatchNorm2d(3)
        self.relu = torch.nn.ReLU()
    def forward(self, x3):
        x3 = self.conv(x3)
        return self.relu(self.bn(x3))
# Inputs to the model
x3 = torch.randn(1, 3, 4, 4)
