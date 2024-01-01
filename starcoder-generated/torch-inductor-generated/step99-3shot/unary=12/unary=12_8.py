
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 8, 3, stride=1, padding=1)
        self.sigmod = torch.nn.Sigmoid()
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1.view(len(x1), -1)
        v3 = torch.sigmoid(v2).view(len(x1), 8, 64 // 3, 64 // 3)
        v4 = v3 * v1
        return v4
# Inputs to the model
x1 = torch.randn(1, 1, 64, 64)
