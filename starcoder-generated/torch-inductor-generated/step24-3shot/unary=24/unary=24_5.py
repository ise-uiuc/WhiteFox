
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv3d(32, 32, (3,3,3), stride=1, padding=(1,1,1))
        self.negative_slope = 0.08
    def forward(self, x):
        v1 = self.conv(x)
        v2 = v1 > 0
        v3 = v1 * self.negative_slope
        v4 = torch.where(v2, v1, v3)
        return v4
# Inputs to the model
x1 = torch.randn(2, 32, 16, 16, 16)
