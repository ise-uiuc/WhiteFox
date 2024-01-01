
class Model(torch.nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size
 
    def forward(self, x):
        x1 = x[:, int(self.size - 1):2 * int(self.size - 1)]
        return torch.cat([x, x1], dim=1)

# Initializing the model
__m__ = Model(1)

# Inputs to the model
__x__ = torch.randn(1, 12)
