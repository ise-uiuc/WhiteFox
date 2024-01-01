
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1):
        v1 = x1.unsqueeze(0).transpose(0, -1).flatten(start_dim=0, end_dim=1)
        v2 = torch.clamp_min(v1, min_value)
        v3 = torch.clamp_max(v2, max_value)
        v4 = v3 / 24.3
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(24, 24)
