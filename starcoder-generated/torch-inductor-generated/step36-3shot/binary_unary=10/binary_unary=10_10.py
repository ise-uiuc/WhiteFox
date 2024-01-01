
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 8)
        self.linear.weight = torch.nn.Parameter(torch.zeros(8, 3))
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 + torch.nn.Parameter(torch.rand(8))
        v3 = torch.relu(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3)
