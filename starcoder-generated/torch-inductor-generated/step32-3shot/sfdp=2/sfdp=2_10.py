
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(10, 5)
        self.linear2 = torch.nn.Linear(5, 1)
 
    def forward(self, x1):
        v0 = x1
        v1 = self.linear1(v0)
        v2 = self.linear2(v1)
        v3 = v2.relu()
        v4 = v3 + 1
        v5 = v2 - v4
        v6 = v5.exp()
        v7 = torch.nn.functional.gelu(v6)
        return v7

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 10)
