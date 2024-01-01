
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pad1 = torch.nn.ReflectionPad1d(13)
        self.conv1 = torch.nn.Conv1d(1, 6, 17, stride=1)
    def forward(self, x):
        negative_slope = -34.230328
        v1 = self.pad1(x)
        v2 = self.conv1(v1)
        v3 = v2 > 0
        v4 = v2 * negative_slope
        v5 = torch.where(v3, v2, v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 1, 85)
