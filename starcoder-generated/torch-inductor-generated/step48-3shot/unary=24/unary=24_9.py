
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = torch.nn.Conv2d(2, 8, 1, stride=1)
    def forward(self, x):
        negative_slope = 0.2607757
        v1 = self.conv2d(x)
        v2 = v1 > 0
        v3 = v1 * negative_slope
        v4 = torch.where(v2, v1, v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 2, 224, 224)
