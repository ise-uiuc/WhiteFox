
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 5)
 
    def forward(self, x1, **kwargs):
        v1 = self.linear(x1)
        v2 = torch.clamp_min(v1, kwargs["v3"])
        v3 = torch.clamp_max(v2, kwargs["v4"])
        return v3

# Initializing the model
kw = {}
kw["v3"] = -1.0
kw["v4"] = 0.0
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3)
