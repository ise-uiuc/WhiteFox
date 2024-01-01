
class Model(torch.nn.Module):
    def __init__(self, negative_slope):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)
        self.negative_slope = negative_slope
 
    def forward(self, x):
        v1 = self.conv(x)
        v2 = v1 > 0
        v3 = v2.float()
        v4 = v1 * v3
        return torch.where(v2, v1, v1 * self.negative_slope)

# Initializing the model
m = Model(0.5)

# Inputs to the model
x = torch.randn(1, 3, 64, 64)
y = torch.randn(1, 3, 64, 64, requires_grad=True)
