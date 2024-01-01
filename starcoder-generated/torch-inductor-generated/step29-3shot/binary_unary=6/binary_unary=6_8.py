
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(5, 2, bias=True)
        self.linear2 = torch.nn.Linear(2, 2, bias=True)
        self.linear3 = torch.nn.Linear(2, 1, bias=True)
 
    def forward(self, x1):
        v1 = self.linear1(x1)
        v2 = v1 - 3
        v3 = torch.relu(v2)
        v4 = self.linear2(v3)
        v5 = v4 - 4.1
        v6 = torch.relu(v5)
        v7 = self.linear3(v6)
        v8 = v7 - 5.2
        return v8

# Initializing the model
m2 = Model()

# Inputs to the model
x2 = torch.randn(1, 5)
