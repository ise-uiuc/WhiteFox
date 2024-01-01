
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1):
        v1 = torch.split(x1, [3, 3], 2)
        v2 = torch.cat([v1[0], v1[1], v1[2]], 2)
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 9, 3)
