
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 10)

    def forward(self, x1):
        v1, v2, v3, v4 = self.linear(x1).chunk(4, 1)

        v5 = v1 * 0.5
        v6 = v1 + (v1 * v1 * v1) * 0.044715
        v7 = v6 * 0.7978845608028654
        v8 = torch.tanh(v7)
        v9 = v8 + 1
        v10 = v5 * v9

        return v10

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
