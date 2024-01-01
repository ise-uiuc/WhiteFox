
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(3, 8, 3, stride=1, padding=1)
 
    def forward(self, x):
        v1 = self.conv(x)
        v2 = torch.gt(v1, 0)
        v3 = v2
        v4 = torch.mul(v1, negative_slope = 0.1)
        v5 = torch.where(v3, v1, v4)
        return v5;

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 3, 64, 64)
