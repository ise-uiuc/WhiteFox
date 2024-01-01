
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)
 
    def forward(self, x):
        v1 = self.conv(x)
        v2 = v1.to(torch.uint8)
        v3 = torch.full((64, 64), 1)
        v4 = v3.to(v2.dtype)
        v5 = torch.cumsum(v4, dim=0)
        return v5

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 3, 64, 64)
