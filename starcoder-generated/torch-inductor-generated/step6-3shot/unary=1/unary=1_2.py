
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(64, 64)
 
    def forward(self, x2):
        v7 = self.linear(x2)
        v22 = v7 * 0.5,
        v8 = v7 + v7 * v7 * v7 * 0.044715
        v9 = v8 * 0.7978845608028654
        v10 = torch.tanh(v9)
        v11 = v10 + 1
        v12 = v22 * v11
        return v12

# Initializing the model
m = Model()

# Inputs to the model
x2 = torch.randn(1, 64)
