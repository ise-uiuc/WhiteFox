
class MatrixMatmulModel(torch.nn.Module):
    def forward(self, arg1, arg2):
        v1 = torch.mm(arg1, arg2)
        v2 = torch.mm(arg1, arg2)
        v3 = v1 + v2
        return v3

# Initializing the model
m = MatrixMatmulModel()

# Inputs to the model
x = torch.randn(20, 10)
y = torch.randn(10, 30)
