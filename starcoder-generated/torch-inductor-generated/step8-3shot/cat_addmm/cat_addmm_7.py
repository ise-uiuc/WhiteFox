
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
     
    def forward(self, x1, x2=torch.randn(1, 3, 32, 32)):
        v1 = torch.addmm(x1, x2, x2)
        v2 = torch.cat([v1], 1)
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
