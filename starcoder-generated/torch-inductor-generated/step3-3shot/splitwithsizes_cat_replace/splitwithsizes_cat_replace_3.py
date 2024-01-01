
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1):
        v1 = x1.clone()
        v2, v3, v4, v5 = torch.split(v1, [1, 2, 3, 4], dim=1)
        return True

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 8, 8, 8)
