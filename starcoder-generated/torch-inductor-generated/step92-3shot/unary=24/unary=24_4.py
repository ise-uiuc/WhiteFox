
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = torch.nn.Conv2d(33, 80, (23, 23), stride=1, padding=(0, 0))
        self.conv1 = torch.nn.Conv2d(80, 56, 1, stride=1, padding=0)
    def forward(self, x):
        negative_slope = -0.25237051
        v1 = self.conv0(x)
        v2 = self.conv1(v1)
        v3 = v2 > 0
        v4 = v2 * negative_slope
        v5 = torch.where(v3, v2, v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 33, 5, 32)
