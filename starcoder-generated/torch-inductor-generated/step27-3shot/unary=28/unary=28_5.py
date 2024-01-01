
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 8)
 
    def forward(self, x1):
        v0 = x1
        v1 = self.linear(v0)
        v2 = torch.clamp_min(v1, min = 0.0)
        v3 = torch.clamp_max(v2, max = 0.4)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 4)
