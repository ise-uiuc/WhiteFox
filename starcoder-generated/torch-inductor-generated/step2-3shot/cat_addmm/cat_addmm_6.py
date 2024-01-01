
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1, x2):
        v1 = torch.addmm(beta=1, input=x1, mat1=x2, mat2=x3)
        v2 = torch.cat([v1], dim=3)
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 5)
x2 = torch.randn(1, 5)
x3 = torch.randn(5, 4)
