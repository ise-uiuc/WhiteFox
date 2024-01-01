
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 16, 1, stride=1, padding=1)
    def forward(self, x1):
        v1 = torch.add(self.conv(x1), 1.0)
        v2 = torch.relu(v1)
        v3 = torch.mul(v2, 0.125)
        v4 = torch.tanh(v3)
        v5 = torch.sigmoid(v4)
        v6 = torch.flatten(v5)
        return v6
# Inputs to the model
x1 = torch.randn(1, 1, 32, 32)
