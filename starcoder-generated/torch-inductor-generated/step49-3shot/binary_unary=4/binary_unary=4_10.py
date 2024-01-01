
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 100)
 
    def forward(self, x1, _other1):
        v1 = self.linear(x1)
        v2 = v1 + _other1
        v3 = F.relu(v2)
        return v3

# Initializng the model
other1 = torch.randn(1, 100)
m = Model()

# Inputs to the model
x1 = torch.randn(1, 10)
