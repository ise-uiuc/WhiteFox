
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)
 
 
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = self.conv2(x)
        v3 = v1 * 0.5
        v4 = v2 * 0.5
        v5 = v1 * 0.7071067811865476
        v6 = v2 * 0.7071067811865476
        v7 = torch.erf(v5)
        v8 = torch.erf(v6)
        v9 = v7 + 1
        v10 = v8 + 1
        v11 = v4 * v9
        v12 = v3 * v10
        v13 = torch.cat([v11, v12], 1)
        return v13

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 3, 64, 64)
