
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 12, 8, stride=3, padding=2)
        self.bn = torch.nn.BatchNorm2d(12)
        self.relu = torch.nn.ReLU()
        self.flatten = torch.nn.Flatten()
    def forward(self, x):
        v1 = self.conv(x)
        v2 = self.bn(v1)
        v3 = self.relu(v2)
        v4 = self.flatten(v3)
        return v4
# Inputs to the model
tensor = torch.randn(64, 1, 224, 224)
