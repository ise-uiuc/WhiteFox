
class Model(torch.nn.Module):
    def __init__(self, x):
        super().__init__()
        self.linear = torch.nn.Linear(x**2, x)
 
    def forward(self, x1, **kwargs):
        v1 = self.linear(x1)
        v2 = v1 + kwargs["x2"]
        v3 = torch.relu(v2)
        return v3

# Initializing the model
m = Model(1)

# Inputs to the model
x1 = torch.randn(1, 1, 2, 2)
x2 = torch.rand(1, 1, 1)
