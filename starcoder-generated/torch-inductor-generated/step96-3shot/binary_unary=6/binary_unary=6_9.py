
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(in_features=9, out_features=1)
 
    def forward(self, x1, x2, x3):
        v1 = self.linear(x1)
        v2 = v1 - x2
        v3 = torch.relu(v2)
        v4 = self.linear(x3)
        v5 = v4 - x2
        v6 = torch.relu(v5)
        v7 = v3 + v6
        return v7
 
# Initializing the model
m = Model()
 
# Inputs to the model
x1 = torch.randn(10, 9, dtype=torch.float)
x2 = torch.randn(1, dtype=torch.float)
x3 = torch.randn(10, 9, dtype=torch.float)
