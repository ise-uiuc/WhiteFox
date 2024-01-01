
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, mat1, mat2, dim):
        v1 = torch.addmm(input, mat1, mat2)
        v2 = torch.cat([v1], dim)
        return v2

# Initializing the model
m = Model()

# Inputs to the model
input = torch.randn(1, 3, 10)
mat1 = torch.rand(6, 6)
mat2 = torch.rand(6, 3)
dim = 0
