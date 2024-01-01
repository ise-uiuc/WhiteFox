
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(16, 8, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 - 1
        v3 = F.relu(v2)
        v4 = torch.stack([v3, v3, v3])
        return v4
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
