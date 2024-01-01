
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0_conv = torch.nn.Conv2d(3, 10, (8, 3), stride=(4, 1), padding=(0, 0))
        self.conv1_conv = torch.nn.Conv2d(10, 8, 3, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv0_conv(x1)
        v2 = v1 * 0.5
        v3 = v1 * 0.7071067811865476
        v4 = torch.erf(v3)
        v5 = v4 + 1
        v6 = v2 * v5
        v7 = self.conv1_conv(v6)
        return v7
# Inputs to the model
x1 = torch.randn(1, 3, 224, 736)
