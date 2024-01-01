
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = torch.nn.Conv2d(3, 3, 1, stride=2, padding=0)
    def forward(self, x):
        negative_slope = -0.81949339
        v1 = self.conv2d(x)
        v2 = v1 > 0        
        v3 = v1 * negative_slope
        v4 = torch.where(v2, v1, v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 26, 35)
