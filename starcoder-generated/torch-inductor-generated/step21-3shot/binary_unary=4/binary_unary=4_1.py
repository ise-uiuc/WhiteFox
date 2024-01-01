
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(20, 40)
        self.linear2 = torch.nn.Linear(40, 50)
        self.linear3 = torch.nn.Linear(50, 10)
 
    def forward(self, x1, other=None):
        v1 = self.linear1(x1)
        v2 = v1 + other if other is not None else v1
        v3 = torch.nn.functional.relu(v2)
        v4 = self.linear2(v3)
        v5 = v4 + other if other is not None else v4
        v6 = torch.nn.functional.relu(v5)
        v7 = self.linear3(v6)
        return v7

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(2, 20)
other = torch.randn(2, 50)
