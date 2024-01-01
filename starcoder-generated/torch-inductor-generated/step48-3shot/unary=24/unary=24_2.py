
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv3d = torch.nn.Conv3d(19, 2, (2, 3, 5), stride=(2, 2, 2))
    def forward(self, x):
        negative_slope = 0.6432200
        v1 = self.conv3d(x)
        v2 = v1 > 0
        v3 = v1 * negative_slope
        v4 = torch.where(v2, v1, v3)
        return v4
# Inputs to the model
x1 = torch.randn(3, 19, 21, 10, 5)
