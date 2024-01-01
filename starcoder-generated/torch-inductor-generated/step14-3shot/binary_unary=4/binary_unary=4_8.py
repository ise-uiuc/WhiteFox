
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(128, 64, bias=True)
        self.linear2 = torch.nn.Linear(64, 32, bias=True)
        self.linear3 = torch.nn.Linear(32, 8, bias=True)
 
    def forward(self, x, other):
        v1 = self.linear1(x)
        v2 = v1 + other
        v3 = F.relu(v2)
        v4 = self.linear2(v3)
        v5 = v4 + other
        v6 = F.relu(v5)
        v7 = self.linear3(v6)
        v8 = v7 + other
        return v8

# Initializing the model
m = Model()

# Inputs to the model
x = torch.empty(1, 128)
other = torch.empty(1, 8)
