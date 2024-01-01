
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(3072, 768, bias=True)
 
    def forward(self, x):
        v0 = torch.flatten(x, 1)
        v1 = self.l1(v0)
        v2 = torch.clamp_min(v1, min=-0.2514298553466797)
        v3 = torch.clamp_max(v2, max=0.5333021664619446)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x_dummy = torch.randn(1, 3, 64, 64)
