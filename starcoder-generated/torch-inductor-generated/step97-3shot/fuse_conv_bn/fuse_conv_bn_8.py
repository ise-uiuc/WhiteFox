
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 3)
        self.bn1 = torch.nn.BatchNorm1d(3)
        self.relu = torch.nn.ReLU()
        self.bn2 = torch.nn.BatchNorm2d(3)
    def forward(self, x4):
        x4 = self.conv(x4)
        return self.bn1(self.relu(x4))
# Inputs to the model
x4 = torch.randn(1, 3, 4, 4)
