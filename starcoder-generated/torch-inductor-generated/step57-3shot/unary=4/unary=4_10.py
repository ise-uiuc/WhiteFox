
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Conv2d(1, 8, 1, stride=1, padding=1)
 
    def forward(self, x2):
        v2 = self.linear(x1)
        v3 = v1 * 0.5
        v4 = v1 * 0.7071067811865476
        v5 = torch.erf(v4)
        v6 = v5 + 1
        v7 = v2 * v6
        return v7

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.rand(1, 3, 64, 64)
