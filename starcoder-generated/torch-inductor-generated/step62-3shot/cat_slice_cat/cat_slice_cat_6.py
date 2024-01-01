
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2, x3, x4):
        v0 = torch.cat([x1, x2, x3, x4], dim=1)
        return v0

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 16, 11, 20)
x2 = torch.randn(1, 16, 11, 20)
x3 = torch.randn(1, 16, 11, 20)
x4 = torch.randn(3, 16, 11, 20)
