
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    # Keyword arguments
    kwargs = {
        "min_value": -0.5, # min_value
        "max_value": 0.5 # max_value
    }

    def forward(self, x1):
        v1 = F.linear(x1, (3072,))
        v2 = torch.clamp_max(v1, 0.9)
        v3 = torch.clamp_max(v2, 0.7)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
