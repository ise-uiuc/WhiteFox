
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2, x3):
        v1 = torch.addmm(x1, x2, x3)
        return torch.cat([v1], dim=1)

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(3, 2, 4)
x2 = torch.randn(2, 4, 5)
x3 = torch.randn(3, 4, 5)

