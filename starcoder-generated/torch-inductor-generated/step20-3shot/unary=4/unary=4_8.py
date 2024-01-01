
## linear_0
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_0 = torch.nn.Linear(4, 4, bias=False)
 
    def forward(self, x1):
        v2 = self.linear_0(x1)
        v3 = v2 * 0.5
        v4 = v2 * 0.7071067811865476
        v5 = torch.erf(v4)
        v6 = v5 + 1
        v7 = v3 * v6
        return v7

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 4)
