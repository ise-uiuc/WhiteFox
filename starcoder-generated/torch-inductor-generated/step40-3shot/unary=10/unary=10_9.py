
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1):
        v1 = x1.flatten(start_dim=1).relu()
        v2 = v1 + 3.
        v3 = v2.clamp(0., 6.)
        v4 = v3 / 6.
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
