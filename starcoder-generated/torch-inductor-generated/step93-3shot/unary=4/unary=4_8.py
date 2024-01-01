
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1000, 2)
 
    def forward(self, x):
        v1 = self.linear(x)
        v2 = v1 * 0.4943
        v3 = v1 * 0.4943
        v4 = torch.erf(v3) - 1
        v5 = v4 + 1
        v6 = v2 * v5
        return torch.sum(v6, dim=1)

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 1000)
