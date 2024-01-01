
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = torch.nn.Conv2d(1, 11, 3, stride=1, padding=1)
        self.conv1d = torch.nn.Conv1d(10, 33, 3, stride=1, padding=1)
    def forward(self, x):
        negative_slope = 0.5882721
        v1 = self.conv2d(x)
        v2 = v1 > 0
        v3 = v1 * negative_slope
        v4 = torch.where(v2, v1, v3)
        v5 = v4.transpose(1, 2)
        v6 = self.conv1d(v5)
        return v6
# Inputs to the model
x1 = torch.randn(1, 1, 39, 56)
