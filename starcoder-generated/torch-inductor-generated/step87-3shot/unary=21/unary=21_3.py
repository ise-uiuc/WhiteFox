
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bn1 = torch.nn.BatchNorm2d(111)
        self.bn2 = torch.nn.BatchNorm2d(111)
    def forward(self, x):
        v1 = self.bn1(x)
        v2 = torch.tanh(v1)
        v3 = self.bn2(v2)
        v4 = torch.tanh(v3)
        return v4
# Inputs to the model
x = torch.randn(1, 111, 128, 64)
