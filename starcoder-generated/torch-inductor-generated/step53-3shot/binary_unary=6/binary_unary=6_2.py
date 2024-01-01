
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x1):
        v1 = x1.flatten()[:, 0]
        v2 = v1 - other
        v3 = v2.reshape(-1, 8, 8, 8)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
