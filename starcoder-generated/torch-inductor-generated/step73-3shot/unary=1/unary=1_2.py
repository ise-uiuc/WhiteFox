
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 8)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 * 0.5
        v5 = (v1 * v1 * v1) * 0.044715
        v3 = v1 + v5
        v4 = v3 * 0.7978845608028654
        v6 = torch.tanh(v4)
        v7 = v6 + 1
        v8 = v2 * v7
        return v8

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3)
