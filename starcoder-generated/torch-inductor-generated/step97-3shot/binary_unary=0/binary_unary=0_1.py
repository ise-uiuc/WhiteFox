
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
    def forward(self, x1, x2):
        v1 = self.conv(x1)
        v2 = v1 + v1
        v3 = torch.relu(v2)
        v4 = v3 + v3 + v1
        v5 = torch.relu(v4)
        v6 = v5 + v2
        v7 = torch.relu(v6)
        return v7 + x2 + v3
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
x2 = torch.randn(1, 16, 64, 64)

