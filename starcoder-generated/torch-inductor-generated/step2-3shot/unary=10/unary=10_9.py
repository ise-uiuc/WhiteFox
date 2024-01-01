
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(30, 75, bias=False)
 
    def forward(self, x):
        v1 = self.linear(x)
        v2 = v1 + 3
        v3 = torch.clamp_min(v2, 0)
        v4 = torch.clamp_max(v3, 6)
        v5 = v4 / 6
        return v5

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 30)
