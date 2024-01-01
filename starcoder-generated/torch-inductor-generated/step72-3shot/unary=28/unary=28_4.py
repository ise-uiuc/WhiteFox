
class Model(torch.nn.Module):
    def __init__(self, min_value=1.0, max_value=2.0):
        super().__init__()
 
    def forward(self, x1):
        v1 = x1.flatten(1)
        v2 = torch.clamp_min(v1, min_value=min_value)
        v3 = torch.clamp_max(v2, max_value=max_value)
        v4 = v3.view_as(x1)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
