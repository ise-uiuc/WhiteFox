
class Model(torch.nn.Module):
    def __init__(self, negative_slope=0.3):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)
        self.negative_slope = negative_slope
 
    def forward(self, x):
        v1 = self.conv(x)
        v2 = torch.gt(v1, 0)
        v3 = v2.type(dtype=v1.dtype)
        v4 = torch.mul(v1, self.negative_slope)
        v5 = torch.where(v2, v1, v4)
        return v5

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 3, 64, 64)
