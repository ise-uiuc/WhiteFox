
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = torch.nn.Conv2d(3, 61, 3, stride=1, padding=4, dilation=1)
    def forward(self, x):
        negative_slope = -1.0660223
        v1 = self.conv2d(x)
        v2 = v1 > 0
        v3 = v1 * negative_slope
        v4 = torch.where(v2, v1, v3)
        return v4
# Inputs to the model
x1 = torch.randn(20, 3, 64, 13) # A specific sized tensor
