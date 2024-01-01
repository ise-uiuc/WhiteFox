
class Model(torch.nn.Module):
    def __init__(self, t1=0.1, t2=-0.1):
        super().__init__()
    
    def forward(self, x1, x1_max=100, x1_min=-100):
        v1 = torch.clamp(x1, x1_min, x1_max)
        v2 = v1 * t1
        v3 = v2 + t2
        return v3

# Initializing the model
m = Model(0.1, -0.1)

# Generating an input tensor
x1 = torch.randn(1, 32, 32)
