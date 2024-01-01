
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 7, stride=1, padding=3)
    def forward(self, x1, x2, x3):
        v1 = self.conv(x1)
        v2 = v1 + x2
        a1 = torch.tanh(v2)
        v3 = v2 + a1
        v4 = a1 + x3
        v5 = torch.tanh(v4)
        v6 = a1 + v5
        v7 = torch.relu(v6)
        return v7
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 3, 64, 64)
x3 = torch.randn(1, 3, 64, 64)
