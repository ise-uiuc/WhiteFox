
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2):
        v1 = x1 + x2
        v2 = x1 + x2
        v3 = torch.cat([v1, v2], 0)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(3, 64)
x2 = torch.randn(3, 64)
