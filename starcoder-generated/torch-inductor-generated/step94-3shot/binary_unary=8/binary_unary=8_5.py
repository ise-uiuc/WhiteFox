
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 12, 5, stride=2, padding=2)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv1(x1)
        v3 = self.conv1(x1)
        v4 = self.conv1(x1)
        v5 = v1 + v2 + v3 + v4
        v6 = torch.tanh(v5)
        return v6
# Inputs to the model
x1 = torch.randn(1, 1, 224, 224)
