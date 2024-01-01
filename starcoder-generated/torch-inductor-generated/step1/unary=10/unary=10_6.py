
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(16, 4)
 
    def forward(self, x):
        v1 = torch.add(self.linear(x), 3)
        v2 = torch.clamp(v1, min=0)
        v3 = torch.clamp(v2, max=6)
        v4 = v3 / 6
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 16)
