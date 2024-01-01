
class Model(torch.nn.Module):
    def __init__(self, min_value, max_value):
        super().__init__()
        self.l = torch.nn.Linear(3, 4)
 
    def forward(self, x1):
        v1 = self.l(x1)
        v2 = torch.clamp_min(v1, min_value=self.kwargs["min_value"])
        v3 = torch.clamp_max(v2, max_value=self.kwargs["max_value"])
        return v3

# Initializing the model
m = Model(min_value=-1.0, max_value=1.0)

# Inputs to the model
x1 = torch.randn(1, 3)

