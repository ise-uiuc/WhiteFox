
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 8)
 
    def forward(self, x):
        v1 = self.linear(x)
        v2 = v1 * 0.5
        v3 = v1 * 0.044715
        v4 = v1 * v1
        v5 = v4 * v1
        v6 = v1 * 0.7978845608028654
        v7 = v3 + v2
        v8 = torch.tanh(v7)
        v9 = v8 + 1
        v10 = v9 * v5
        v11 = v10 * v6
        return v11

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 8)
