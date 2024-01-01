
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(6, 1)
        self.linear2 = torch.nn.Linear(6, 1)
        self.linear3 = torch.nn.Linear(6, 1)
 
    def forward(self, x1, other):
        v1 = self.linear1(x1)
        v2 = v1 + other
        v3 = self.linear2(x1)
        v4 = v3 + other
        v5 = self.linear3(v4)
        return v5

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(5, 6)
other = torch.randn(5, 1)
