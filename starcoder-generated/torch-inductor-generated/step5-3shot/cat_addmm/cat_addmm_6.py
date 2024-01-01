
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x5, x8):
        v1 = torch.addmm(x1, x5, x8)
        v2 = torch.cat([v1], 1)
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 14)
x5 = torch.randn(14, 256)
x8 = torch.randn(256, 256)
