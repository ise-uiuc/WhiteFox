
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(16, 16, 5, stride=2, padding=2)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 + v1
        v3 = torch.relu(v2)
        v4 = torch.tanh(v3)
        v5 = v3 + v4
        v6 = torch.relu(v5)
        v7 = torch.tanh(v6)
        return v7
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
