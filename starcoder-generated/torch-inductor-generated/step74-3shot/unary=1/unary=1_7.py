
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 16)
 
    def forward(self, x2):
        v7 = self.linear(x2)
        v8 = v7 * 0.5
        v9 = v7 + torch.pow(v7, 2 * 3) * 0.044715
        v10 = v9 * 0.7978845608028654
        v11 = torch.tanh(v10)
        v12 = v11 + 1
        v13 = v8 * v12
        return v13

# Initializing the model
m = Model()

# Inputs to the model
x2 = torch.randn(1, 10)
