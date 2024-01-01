
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
 
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.cat([v1, v1, v1], dim=1)
        v3 = v2[:, 0:1073741823]
        v4 = v2[:, 0:2147483647]
        v5 = torch.cat([v2, v3, v4], dim=1)
        return v5

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
