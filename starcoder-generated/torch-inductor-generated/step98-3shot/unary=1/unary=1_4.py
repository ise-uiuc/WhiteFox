
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(64, 64)
 
    def forward(self, x1):
        v1 = torch.tanh(self.linear(x1))
        v2 = v1 + (v1 * v1 * v1) * 0.044715
        v3 = v2 * 0.7978845608028654
        v4 = torch.tanh(v3)
        v5 = v4 + 1
        v6 = v1 * v5
        return v6

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 64)
