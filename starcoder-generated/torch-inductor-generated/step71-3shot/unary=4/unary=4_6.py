
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)
        self.linear = torch.nn.Linear(1024, 32)
 
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.linear(v1)
        v3 = v2 * 0.5
        v4 = self.linear(v2)
        v5 = v4 * 0.7071067811865476
        v6 = torch.erf(v5)
        v7 = v6 + 1
        v8 = v3 * v7
        return v8

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 1024, 1, 1)
