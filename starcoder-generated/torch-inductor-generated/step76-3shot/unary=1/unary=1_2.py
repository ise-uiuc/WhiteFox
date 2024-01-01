
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(8, 5)
        self.linear2 = torch.nn.Linear(5, 4)
        self.linear3 = torch.nn.Linear(4, 3)
 
    def forward(self, x3):
        v1 = self.linear1(x3)
        v2 = v1 * 0.5
        v3 = v1 + (v1 * v1 * v1) * 0.044715
        v4 = v3 * 0.7978845608028654
        v5 = torch.tanh(v4)
        v6 = v5 + 1
        v7 = v2 * v6
        v8 = self.linear2(v7)
        v9 = self.linear3(v8)
        return v9

# Initializing the model
m = Model()

# Inputs to the model
x3 = torch.randn(1, 8)
