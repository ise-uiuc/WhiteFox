
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2, b1, b2):
        v1 = torch.addmm(x1, b1, x2) 
        v2 = torch.cat([v1], dim=b2.dim() - 1)
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 256, 1024)
x2 = torch.randn(1, 12, 256, 1024)
b1 = torch.randn(1, 256, 12)
b2 = 3
