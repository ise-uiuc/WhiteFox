
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 5, stride=1, padding=2, bias=False)
    def forward(self, x1):
        v0 = x1.sum(dim=(1, 2, 3))
        v0 = v0.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        v1 = self.conv(v0)
        v2 = self.conv(v0)
        v3 = self.conv(x1)
        v4 = self.conv(x1)
        v5 = torch.cat([v1, v2, v3, v4], dim=1)
        v6 = v5 + v0
        v7 = torch.relu(v6)
        return v7
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
