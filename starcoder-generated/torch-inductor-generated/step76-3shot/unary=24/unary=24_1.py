
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv1d(7, 3, 3, stride=3)
    def forward(self, x):
        negative_slope = 2.257444
        v1 = self.conv(x).transpose(1, 2)
        v2 = v1 > 0
        v3 = v1 * negative_slope
        v4 = torch.where(v2, v1, v3).transpose(1, 2)
        return v4
# Inputs to the model
x1 = torch.randn(3, 7, 16)
