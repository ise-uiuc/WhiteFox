
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(64, 32, bias=True)
 
    def forward(self, x1, x2):
        # x1.shape = [x1]
        d1 = torch.add(x1, x1, alpha=1)
        d2 = torch.add(x1, x2, alpha=1)
        d3 = torch.add(x2, x1, alpha=1)
        # d4.shape = [d4]
        v1 = self.conv(x1)
        v2 = v1 * 0.5
        v3 = v1 * 0.7071067811865476
        v4 = torch.erf(v3)
        v5 = v4 + 1
        v6 = v2 * v5
        return v6

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 3, 64, 64)
x1, x2, d4 = m(x1, x2)
