
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(32, 32)
 
    def forward(self, x1, min_v, max_v):
        v1 = self.lin(x1)
        v2 = torch.clamp_min(v1, min_v)
        v3 = torch.clamp(v2, None, max_v)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 32)
min_v = 0.2
max_v = 0.8
