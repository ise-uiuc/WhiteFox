
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 2)
        self.bn = torch.nn.BatchNorm2d(3)
        self.relu = torch.nn.ReLU()
    def forward(self, x5):
        x5 = self.conv(x5)
        return self.relu(self.bn(x5))
# Inputs to the model
x5 = torch.randn(1, 3, 4, 4)
