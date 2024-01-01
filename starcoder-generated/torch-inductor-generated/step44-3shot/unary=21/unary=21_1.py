
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 1, kernel_size=5, padding=5, bias=False)
        self.conv2 = torch.nn.Conv2d(1, 3, kernel_size=5, padding=5, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(1)
        self.bn2 = torch.nn.BatchNorm2d(3)
    def forward(self, x):
        v1 = self.bn1(x)
        v2 = self.conv1(v1)
        v3 = torch.tanh(v2)
        v4 = self.bn2(v3)
        v5 = self.conv2(v4)
        return torch.tanh(v5)
# Inputs to the model
x = torch.randn(1, 3, 224, 224)
