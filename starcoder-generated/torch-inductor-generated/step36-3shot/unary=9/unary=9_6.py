
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = F.pad(v1, (1, 1, 1, 1), "constant", 3)
        v3 = F.relu6(v2)
        v4 = F.normalize(v3, p=6, dim=1)
        return v4
# Inputs to the model
x1 = torch.randn(5, 3, 64, 64)
