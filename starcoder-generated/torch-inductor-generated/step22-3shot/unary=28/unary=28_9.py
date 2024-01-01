
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, min_value = 0.2, max_value = 0.1):
        v1 = torch.nn.functional.linear(x1, torch.randn(10, 30))
        v2 = torch.clamp_min(v1, min_value = min_value)
        v3 = torch.clamp_max(v2, max_value = max_value)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 10)
