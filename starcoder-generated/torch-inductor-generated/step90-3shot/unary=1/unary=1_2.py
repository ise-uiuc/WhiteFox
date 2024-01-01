
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(16, 16)
        self.linear2 = torch.nn.Linear(16, 8)
        self.linear3 = torch.nn.Linear(8, 4)
 
    def forward(self, x1):
        v1 = self.linear1(x1)
        v2 = v1 * 0.5
        v3 = self.linear2(v1*v1*v1)
        v4 = v3 * 0.044715
        v5 = v3 * 0.7978845608028654
        v6 = torch.tanh(v5)
        v7 = v6 + 1
        v8 = v2 * v7
        v9 = v1 * v8
        v10 = self.linear3(v9)
        return v10

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 16)
