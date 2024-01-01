
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)
 
    def forward(self, x):
        v1 = self.conv(x)
        v2 = v1 + 3
        v3 = v2.clamp_min_(0)
        v4 = v3.clamp_max_(6)
        v5 = v4 * v1
        v6 = v5 / 6
        return v6

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 3, 64, 64)
