
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 6)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1
        v3 = torch.clamp(min=0.0, max=6.0, v2)
        v4 = v3 + 3
        v5 = v4 / 6
        return v5

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3)
