
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 8)
 
    def forward(self, x4):
        v7 = self.linear(x4)
        v8 = v7 * 0.5
        v10 = v7 + (v7 * v7 * v7) * 0.044715
        v13 = v10 * 0.7978845608028654
        v16 = torch.tanh(v13)
        v19 = v16 + 1
        v22 = v8 * v19
        return v22

# Initializing the model
m = Model()

# Inputs to the model
x4 = torch.randn(1, 1)
