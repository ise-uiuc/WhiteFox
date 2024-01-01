
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 2, stride=2, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.conv(v1)
        v3 = self.conv(v2)
        v4 = v1 + v2 + v3
        v5 = v2 + v3 + v4
        v6 = [v5, v4]
        v7 = v6[1:]
        v8 = torch.relu(v6)
        return v7
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
