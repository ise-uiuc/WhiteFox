 (cont'd.)
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mat1 = torch.nn.Parameter(torch.randn(256, 64, 7, 7))
        self.mat2 = torch.nn.Parameter(torch.randn(256, 64, 1, 1))
 
    def forward(self, x1):
        v1 = torch.addmm(x1, self.mat1, self.mat2)
        v2 = torch.cat([v1], dim = 1)
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 64, 28, 28)
