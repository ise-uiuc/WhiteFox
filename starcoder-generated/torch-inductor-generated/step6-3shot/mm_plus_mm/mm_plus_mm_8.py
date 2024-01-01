
class Model(torch.nn.Module):
    def forward(self, x1, x2, z1, z2):
        b1 = torch.matmul(x1, x2)
        b2 = torch.matmul(z1, z2)
        c = b1 + b2
        return c
# Inputs to the model
x1 = torch.randn(3, 3)
x2 = torch.randn(3, 3)
z1 = torch.randn(3, 3)
z2 = torch.randn(3, 3)
