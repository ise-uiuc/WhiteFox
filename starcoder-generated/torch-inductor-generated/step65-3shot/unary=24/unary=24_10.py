
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 7, kernel_size=(10, 4), stride=4, padding=(0, 3))
    def forward(self, x):
        negative_slope = 1.1830052
        v1 = self.conv(x)
        v2 = v1 > 0
        v3 = v1 * negative_slope
        v4 = torch.where(v2, v1, v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 1, 21, 81)
