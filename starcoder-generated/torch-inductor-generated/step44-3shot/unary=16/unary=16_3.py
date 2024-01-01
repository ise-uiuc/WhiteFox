
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
 
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 * 0.5
        v3 = v1 * 0.7071067811865476
        v4 = torch.erf(v3)
        v5 = v4 + 1
        v6 = v2 * v5
        t1 = self.linear(v6)
        v9  = t1 * 0.5
        v10 = t1 * 0.7071067811865476
        v11 = torch.erf(v10)
        v12 = v11 + 1
        v13 = v9 * v12
        t2 = self.t2 = relu(v13)

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
