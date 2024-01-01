
class Model(torch.nn.Module):
    def forward(self, x, y):
        x = torch.Tensor.addmm(b=x, mat1=y, mat2=y, alpha=1, beta=1)
        return x

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(8, 8)
y = torch.randn(8, 8)
