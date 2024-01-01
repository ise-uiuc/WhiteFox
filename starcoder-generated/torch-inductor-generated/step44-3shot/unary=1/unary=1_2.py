
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 * 0.5
        v3 = v1
        v4 = v3 * v3 * v3
        v5 = v3 + (v4 * 0.044715)
        v6 = v5 * 0.7978845608028654
        v7 = torch.tanh(v6)
        v8 = v7 + 1
        v9 = v2 * v8
        return v9

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1)
