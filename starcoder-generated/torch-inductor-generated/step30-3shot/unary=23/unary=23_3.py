
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = torch.nn.BatchNorm2d(2)
        self.relu = torch.nn.ReLU6()
    def forward(self, x1):
        v1 = self.bn(x1)
        v2 = self.relu(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 2, 14, 28)
