
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(64, 8)
 
    def forward(self, x):
        v1 = self.linear(x)
        v2 = v1 + 3
        v3 = torch.clamp(v2, min=0, max=6)
        v4 = v1 * v3
        return v4 / 6

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 64)
