
class Model(torch.nn.Module):
    def forward(self, x, y, z):
        v1 = x.matmul(y)
        v2 = z.matmul(v1)
        return z + v2
# Inputs to the model
x = torch.randn(20, 20)
y = torch.randn(20, 20)
z = torch.randn(20, 20)
