
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(5, 4, 3, stride=1, padding=1)
        self.norm = torch.nn.BatchNorm2d(4)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.norm(v1)
        v3 = F.sigmoid(v2)
        v4 = v2 * v3
        return v4
# Inputs to the model
x1 = torch.randn(1, 5, 64, 64)
