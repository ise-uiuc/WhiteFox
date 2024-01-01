
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(34,112)
        self.linear2 = torch.nn.Linear(112,68)
        self.linear3 = torch.nn.Linear(68,3)
 
    def forward(self, x1):
        v1 = self.linear1(x1)
        v2 = v1 - 1
        v3 = torch.nn.functional.relu(v2)
 
        v4 = self.linear2(v3)
        v5 = v4 - 2
        v6 = torch.nn.functional.relu(v5)
 
        v7 = self.linear3(v6)
        v8 = v7 - 3
        return v8
 
# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 34)
