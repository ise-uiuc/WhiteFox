
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)
 
    def forward(self, x):
        v1 = self.conv(x)
        v2 = torch.gt(v1, 0)
        v3 = torch.where(v2, v1, v1 * 0.2)
        return v3

# Initializing the model
m = Model(negative_slope=0.2)

# Inputs to the model
x = torch.randn(1, 3, 64, 64)
