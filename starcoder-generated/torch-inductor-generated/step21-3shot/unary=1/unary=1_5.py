
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(300000, 10)
 
    def forward(self, x1):
        v3 = self.linear(x1)
        v5 = v3 * 0.5
        v6 = v3 + (torch.pow(v3, 3)) * 0.044715
        v7 = v6 * 0.7978845608028654
        v8 = torch.tanh(v7)
        v9 = v8 + 1
        v10 = v5 * v9
        return v10

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 300000)
