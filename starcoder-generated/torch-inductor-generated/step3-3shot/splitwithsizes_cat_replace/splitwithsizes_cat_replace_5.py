
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 1, stride=1, padding=1)
 
    def forward(self, x1):
        v1, v2, v3 = torch.split(x1, 2, 1)
        v4 = x1.permute(0, 2, 1)
        v5, v6, v7 = torch.split(v4, 2, 1)
        v8 = v1.permute(0, 2, 1)
        v9 = torch.cat((v2, v3, v5, v6, v7, v8), 1)
        return v9

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
