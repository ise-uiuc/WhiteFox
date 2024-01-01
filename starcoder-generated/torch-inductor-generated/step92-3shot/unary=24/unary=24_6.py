
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = torch.nn.Conv2d(9, 71, 1, stride=2, padding=0)
        self.conv1 = torch.nn.Conv2d(71, 48, (4, 1), stride=4, padding=(3, 0))
        self.conv2 = torch.nn.Conv2d(48, 73, (1, 1), stride=1, padding=0)
    def forward(self, x):
        negative_slope = 0.17195540307039802
        v1 = self.conv0(x)
        v2 = self.conv1(v1)
        v3 = self.conv2(v2)
        v4 = v3 > 0
        v5 = v3 * negative_slope
        v6 = torch.where(v4, v3, v5)
        return v6
# Inputs to the model
x1 = torch.randn(1, 9, 78, 25)
