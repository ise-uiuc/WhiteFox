
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(15, 32, bias=False)
        self.linear2 = torch.nn.Linear(32, 2, bias=False)
 
    def forward(self, x):
        v1 = self.linear1(x)
        v2 = v1 * 0.5
        v3 = self.linear1(x)
        v4 = v3 * v3
        v5 = v4 * v4
        v6 = v5 * 0.044715
        v7 = self.linear1(x) * v6
        v8 = v7 * 0.7978845608028654
        v9 = v8  + 1
        v10 = torch.tanh(v9)
        v11 = v7 + v10
        v12 = v2 * v11
        return v12

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 15)
