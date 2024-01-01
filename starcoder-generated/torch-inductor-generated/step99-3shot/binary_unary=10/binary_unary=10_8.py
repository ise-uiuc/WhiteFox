
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(4, 8)
        self.linear2 = torch.nn.Linear(8, 8)
        self.linear3 = torch.nn.Linear(8, 4)
 
    def forward(self, x1, x2):
        v1 = self.linear1(x1)
        v2 = x2 + v1
        v3 = v2.relu()
        v4 = self.linear2(v3)
        v5 = self.linear3(v4)
        return v5

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 4)
x2 = torch.randn(1, 8)
