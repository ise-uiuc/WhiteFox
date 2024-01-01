
class Model(torch.nn.Module):
    def __init__(self, min_value = 0, max_value = 6):
        super().__init__()
    
    def forward(self, x1):
        v1 = x1.flatten()
        v2 = torch.clamp_min(v1, 0)
        v3 = torch.clamp_max(v2, 6)
        v4 = v3.reshape(x1.shape)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(128, 128)
