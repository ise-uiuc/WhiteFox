
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
 
    def forward(self, x1, x2, x3):
        x = torch.cat([x1, x2, x3], dim=1)
        v1 = x[:, 0:torch.iinfo(torch.int64).max]
        v2 = v1[:, 0:5]
        x2 = torch.cat([x, v2], dim=1)
        return x2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
x2 = torch.randn(1, 3, 32, 32)
x3 = torch.randn(1, 3, 32, 32)
