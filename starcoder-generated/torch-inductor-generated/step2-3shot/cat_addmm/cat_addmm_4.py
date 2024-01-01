
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, input, mat1, mat2, dim=1):
        v1 = torch.addmm(input, mat1, mat2)
        v2 = torch.cat([v1], dim)
        return v2

# Initializing the model
m = Model()

# Inputs to the model
input = torch.randn(2, 2, 4096, 1024)
mat1 = torch.randn(2, 1024, 512)
mat2 = torch.randn(2, 1024, 512)
dim = 1
