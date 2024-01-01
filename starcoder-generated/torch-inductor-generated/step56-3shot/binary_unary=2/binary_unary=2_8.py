
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(32, 32, 1, stride=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = v1 - -0.3
        v3 = v2 + 0.1
        v4 = v3 + 0.01
        v5 = v4 - 100.1
        return torch.mm(v5, v5)
# Inputs to the model
x1 = torch.randn(1, 32, 8, 8)
