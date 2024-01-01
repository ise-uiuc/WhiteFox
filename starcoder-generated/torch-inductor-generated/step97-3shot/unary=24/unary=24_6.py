
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(6, 5, 1, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(5, 6, 3, stride=2, padding=0)
    def forward(self, x):
        negative_slope = 2.044072
        v1 = self.conv1(x)
        v2 = self.conv2(v1)
        v3 = v2 > 0
        v4 = v2 * negative_slope
        v5 = torch.where(v3, v2, v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 6, 51, 1)
