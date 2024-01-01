
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, mat1, mat2):
        v1 = torch.addmm(input, mat1, mat2)
        t2 = torch.cat([v1], dim)
        return t2

# Initializing the model
m = Model()

# Inputs to the model
input = torch.randn(1, 64, 64)
mat1 = torch.randn(1, 64, 32)
mat2 = torch.randn(1, 64, 32)
dim = 1
