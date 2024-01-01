
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(64, 16)
        self.linear.bias.data = torch.zeros_like(self.linear.bias.data)
        self.linear.weight.data = torch.zeros_like(self.linear.weight.data)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 + 0.5
        v3 = v2 + 0.7071067811865476
        v4 = torch.erf(v3)
        v5 = v4 + 1
        v6 = v2 * v5
        return v6

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 64)
