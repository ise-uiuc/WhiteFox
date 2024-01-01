
class Model(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x = torch.addmm(input, mat1, mat2)
        x = x - x
        x = x + x
        x = torch.addmm(input, mat1, mat2)
        return x
# Inputs to the model
input = torch.randn(2, 5)
mat1 = torch.randn(5, 12)
mat2 = torch.randn(12, 8)
