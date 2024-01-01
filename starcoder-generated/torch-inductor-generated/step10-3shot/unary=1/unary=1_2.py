
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(32, 16)
 
    def forward(self, x2):
        v2 = self.linear(x2)
        v3 = v2 * 0.5
        v4 = v2 + (v2 * v2 * v2) * 0.044715
        v5 = v4 * 0.7978845608028654
        v6 = torch.tanh(v5)
        v7 = v6 + 1
        v8 = v3 * v7
        return v8

# Initializing the model
m = Model()

# Inputs to the model
x2 = torch.randn(1, 32)
