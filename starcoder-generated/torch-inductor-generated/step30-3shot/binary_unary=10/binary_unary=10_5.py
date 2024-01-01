
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(1600, 120)
        self.linear2 = torch.nn.Linear(120, 84)
        self.linear3 = torch.nn.Linear(84, 10)
 
    def forward(self, x, other):
        v1 = self.linear1(x)
        v2 = v1 + other
        v3 = relu(v2)
        v4 = self.linear2(v3)
        v5 = v4 + other
        v6 = relu(v5)
        v7 = self.linear3(v6)
        return v7

# Initializing a model
m = Model()

# Inputs to the model
x = torch.randn(1, 1600)
other = torch.randn(1, 120)
