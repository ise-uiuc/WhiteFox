s
class Model_min_max(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1):
        v1 = torch.nn.functional.linear(x1, torch.zeros(5,3))
        v2 = torch.clamp_min(v1, min=-1.0)
        v3 = torch.clamp_max(v2, max=1.0)
        return v3

# Initializing the model
m = Model_min_max()

# Inputs to the model
x1 = torch.randn(1, 3)
