
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, 3, stride=1, padding=1)
    def forward(self, x):
        negative_slope = -0.001
        v1 = self.conv(x)
        v2 = v1 > 2
        v3 = v1 * 0
        v4 = torch.where(v2, v1, v3)
        v5 = v4 * -1 
        v6 = v4 + v5  
        return v6
# Inputs to the model
x1 = torch.randn(2, 1, 64, 64)
