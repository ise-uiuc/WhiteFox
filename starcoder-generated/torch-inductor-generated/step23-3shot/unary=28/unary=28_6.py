
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8192, 147)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.clamp_min(v1, min=0.151, max_value=None)
        v3 = torch.clamp_max(v2, max=0.248, min_value=None)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 8192)
