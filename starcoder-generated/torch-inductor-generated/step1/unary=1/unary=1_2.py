
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 2)
 
    def forward(self, x):
        v1 = self.linear(x)
        v2 = v1 * 0.5
        v3 = v1 * 0.7978845608028654
        v4 = v2 + v3
        v5 = torch.tanh(v4)
        v6 = v5 + 1
        v7 = v1 + v1
        v8 = v6 * v6
        v9 = v7 * v7
        v10 = v6 * v8
        return v10

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 8)
