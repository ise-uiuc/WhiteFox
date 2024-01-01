
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(128, 64)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 * 0.11753406086160044
        v3 = v1 * 0.11624760337444353
        v4 = torch.erf(v3)
        v5 = v4 + 1
        v6 = v2 * v5
        return v6

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(4, 128)
