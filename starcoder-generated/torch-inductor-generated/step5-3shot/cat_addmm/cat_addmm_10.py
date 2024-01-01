
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, mat1, mat2):
        return torch.cat([torch.addmm(input, mat1, mat2)], dim=1)

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 16)
mat1 = torch.randn(4, 16)
mat2 = torch.randn(4, 16)
