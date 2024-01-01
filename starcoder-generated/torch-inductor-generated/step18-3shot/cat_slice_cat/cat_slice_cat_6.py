
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2, x3):
        v1 = [x1, x2, x3]
        v2 = torch.cat(v1, dim=1)
        v3 = v2[:, 0:-1]
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 8, 64, 64)
x2 = torch.randn(1, 4, 64, 64)
x3 = torch.randn(1, 32, 64, 64)
