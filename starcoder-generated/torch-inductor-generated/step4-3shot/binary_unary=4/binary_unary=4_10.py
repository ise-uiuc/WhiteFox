
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.other = torch.nn.Parameter(torch.ones((10)))
        self.linear = torch.nn.Linear(10, 20)
 
    def forward(self, x, other=None):
        v1 = self.linear(x)
        v2 = v1 + other
        v3 = F.relu(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(2, 10)
