
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
 
    def forward(self, x1):
        v1, v2 = torch.split(x1, [3,3], 1)
        v3, v4 = torch.split(x1, [3,3], 1)
        v5, v6 = torch.split(x1, [2,4], 2)
        v7 = torch.cat([v1, v3, v5], 1)
        v8 = torch.cat([v2, v4, v6], 1)
        return v7, v8

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 6, 64, 64)
