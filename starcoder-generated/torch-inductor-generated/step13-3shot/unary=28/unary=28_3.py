
class Model(torch.nn.Module):
    def __init__(self, min=0, max=1):
        super().__init__()
        self.linear = torch.nn.Linear(3, 4)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.clamp_min(v1, min=min)
        v3 = torch.clamp_max(v2, max=max)
        return v3

# Initializing the model
m = Model(min=0, max=1)

# Inputs to the model
x1 = torch.randn(2, 3)
