
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(32, 16)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 + torch.ones(1, 16)
        v3 = torch.nn.functional.relu(v2)
        return torch.sum(v3)

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(2, 32)
__output = m(x1)

