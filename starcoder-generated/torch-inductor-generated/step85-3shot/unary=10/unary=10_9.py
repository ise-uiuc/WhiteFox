
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1):
        v1 = torch.clamp_min(x1, 0)
        v2 = torch.clamp_max(v1, 6)
        v3 = v2 / 6
        return v3
 
# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
