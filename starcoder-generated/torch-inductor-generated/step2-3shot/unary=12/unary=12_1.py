
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = F.sigmoid(v1)
        v3 = F.sigmoid(v1)
        v4 = v1 * v3
        v5 = v1 * v2
        v6 = v4 * v5
        return v6
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
