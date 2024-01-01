
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Linear(32, 3)
 
    def forward(self, x):
        v1 = self.net(x)
        v2 = v1 + 3
        v3 = torch.clamp(v2, min=0)
        v4 = torch.clamp(v3, max=6)
        v5 = v4 / v4
        return v5

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 32)
