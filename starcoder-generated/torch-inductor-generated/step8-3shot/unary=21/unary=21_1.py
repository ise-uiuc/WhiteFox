
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 7, stride=5, padding=3)
        self.bn = torch.nn.BatchNorm2d(8)
        self.relu = torch.nn.ReLU()
    def forward(self, x):
        v1 = self.conv(x)
        v2 = self.bn(v1)
        v3 = self.relu(v2)
        v4 = torch.tanh(v3)
        return v4
# Inputs to the model
x = torch.randn(64, 3, 64, 64)
