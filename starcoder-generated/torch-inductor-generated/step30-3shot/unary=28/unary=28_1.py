
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
 
 
    def forward(self, x1):
        v1 = torch.nn.functional.linear(x1, weight, None)
        v2 = torch.clamp(v1, min, max)
        return v2

# Initializing the model
m = Model(min)

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
