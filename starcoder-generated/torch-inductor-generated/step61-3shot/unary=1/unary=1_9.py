
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(32, 5)
 
    def forward(self, x0):
        v0 = self.linear(x0)
        v1 = v0 * 0.5
        v2 = v0 + (v0 * (v0 * v0)) * 0.044715
        v3 = v2 * 0.7978845608028654
        v4 = torch.tanh(v3)
        v5 = v4 + 1
        v6 = v1 * v5
        return v6

# Initializing the model
m = Model()

# Inputs to the model
x0 = torch.randn(1, 32)
