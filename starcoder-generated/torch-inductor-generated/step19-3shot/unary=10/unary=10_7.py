
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 3)
 
    def forward(self, x):
        v = self.linear(x)
        v1 = v + 3
        v2 = torch.clamp_min(v1, 0)
        v3 = torch.clamp_max(v2, 6)
        v4 = v3 / 6
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 3)
