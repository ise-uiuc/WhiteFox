
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 16, 15, stride=1, padding=11)
        self.sigmoid1 = torch.nn.Sigmoid()
        self.sigmoid2 = torch.nn.Sigmoid()
    def forward(self, x1):
        v0 = x1
        v1 = self.conv(v0)
        v2 = self.sigmoid1(v1)
        v3 = self.sigmoid2(v2)
        v4 = v2 * v3
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
