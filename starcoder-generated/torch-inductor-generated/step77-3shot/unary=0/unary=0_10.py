
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(12, 37, 1, stride=1, padding=0)
        self.conv_0 = torch.nn.Conv2d(37, 12, 5, stride=4, padding=3)
        self.conv_0_0 = torch.nn.Conv2d(12, 12, 3, stride=2, padding=1)
        self.conv_0_0_0 = torch.nn.Conv2d(12, 12, 5, stride=2, padding=3)
    def forward(self, x198):
        v1 = self.conv(x198)
        v2 = self.conv_0(v1)
        v3 = self.conv_0_0(v1)
        v4 = self.conv_0_0_0(v3)
        v5 = v1 + v4
        v6 = v2 * v5
        return v6
# Inputs to the model
x198 = torch.randn(1, 12, 4, 400)
