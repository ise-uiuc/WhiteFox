
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)
 
    def forward(self, x, negative_slope):
        v1 = self.conv(x)
        v2 = v1 > 0
        v3 = torch.zeros_like(v1)
        v4 = torch.where(v2, v1, v3)
        v5 = v4 * negative_slope
        return v5

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 3, 64, 64)
negative_slope = 1e-2
