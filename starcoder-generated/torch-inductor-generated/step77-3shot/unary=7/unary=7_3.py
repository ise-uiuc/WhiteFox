
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 8)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 * torch.clamp(torch.relu(v1 + 3), min=0, max=6)
        v3 = v2 / 6
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 8, 8)
x2 = torch.ones(1, 3, 8, 8)
x3 = torch.zeros(1, 3, 8, 8)
outputs = torch.cat([m(x1), m(x2), m(x3)], dim=-1)

