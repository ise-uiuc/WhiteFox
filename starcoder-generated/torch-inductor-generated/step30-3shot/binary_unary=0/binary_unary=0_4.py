
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(32, 16, 3, stride=2, padding=1)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = 1 + v1
        v3 = torch.tanh(v2)
        v4 = 1 + 1
        v5 = v3 + v4
        v6 = v5 * v1
        v7 = v6.view(16)
        v8 = v7 - v7
        v9 = torch.sigmoid(v8)
        return v9
# Inputs to the model
x = torch.randn(1, 32, 28, 28)
