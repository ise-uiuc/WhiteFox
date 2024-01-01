
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 8)
 
    def forward(self, x0):
        v1 = self.linear(x0)
        v2 = torch.clamp_min(v1, min=0)
        v3 = torch.clamp_max(v2, max=6)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x0 = torch.randn(1, 3)
