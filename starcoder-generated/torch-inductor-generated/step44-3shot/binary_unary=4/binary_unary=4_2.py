
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(24, 24)
 
    def forward(self, x1, **kwargs):
        v1 = self.linear(x1)
        return torch.add(v1, kwargs["v1"])

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 24)
x2 = torch.randn(1, 24)
