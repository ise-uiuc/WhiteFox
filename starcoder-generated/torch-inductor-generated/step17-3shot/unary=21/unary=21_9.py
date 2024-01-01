
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 10, 5, stride=3, bias=True)
        self.bn = torch.nn.BatchNorm2d(10)
        self.relu = torch.nn.ReLU()
    def forward(self, x):
        v1 = self.conv(x)
        v2 = self.bn(v1)
        v2 = self.relu(v2)
        return torch.tanh(v2)
# Inputs to the model
x = torch.randn(1, 1, 100, 32)
