
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(16, 32)
        self.linear2 = torch.nn.Linear(32, 32)
        self.linear3 = torch.nn.Linear(32, 64)
 
    def forward(self, x, other):
        v1 = self.linear(x)
        v2 = v1 + other
        v3 = F.relu(v2)
        v4 = self.linear2(v3)
        v5 = v4 + other
        v6 = F.relu(v5)
        v7 = self.linear3(v6)
        return v7

# Initializing the model   
m = Model()

# Inputs to the model
x = torch.randn(1, 16)
other = torch.randn(1, 32)
