
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 8)
 
    def forward(self, x):
        v1 = self.linear(x)
        v2 = v1 + 3
        v4 = torch.clamp_min(v2, 0)
        v5 = torch.clamp_max(v4, 6)
        v6 = v5 / 6
        return v6

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 2)
