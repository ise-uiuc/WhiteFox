
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(16, 32)
 
    def forward(self, x1):
        flattened = torch.flatten(x1, 1)
        v1 = self.linear(flattened)
        v2 = torch.clamp_min(v1, min=0)
        v3 = torch.clamp_max(v2, max=255)  # Note that clamp_max is inclusive
        reshape = v3.reshape(1, 16, 8, 8)
        sigmoid = torch.sigmoid(reshape)
        return sigmoid

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 16, 2, 2)
