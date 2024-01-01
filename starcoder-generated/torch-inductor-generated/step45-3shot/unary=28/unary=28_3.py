
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 16)
 
    def forward(self, x1):
        v1 = self.linear(x)
        v2 = torch.clamp_min(v1, min_value=0., max_value=None)
        v3 = torch.clamp_max(v2, min_value=None, max_value=0.5)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 8)
