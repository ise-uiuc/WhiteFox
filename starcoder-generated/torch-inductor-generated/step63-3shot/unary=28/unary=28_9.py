
class Model(torch.nn.Module):
    def __init__(self, min_value: float, max_value: float):
        super().__init__()
 
    def forward(self, x1, x2, x3):
        v1 = x1 * x2
        v2 = x3 * x4
        return torch.clamp(v2, min=min_value, max=max_value)

# Initializing the model
m = Model(min_value=0.3, max_value=0.6)

# Inputs to the model
x1 = torch.randn(2, 16, 3, 128, 128)
x2 = torch.randn(2, 16, 3, 128, 128)
x3 = torch.randn(2, 16, 3, 128, 128)
