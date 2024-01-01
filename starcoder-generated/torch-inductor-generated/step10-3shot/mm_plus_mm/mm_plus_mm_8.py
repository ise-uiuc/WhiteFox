
class Model(torch.nn.Module):
    def forward(self, x1, x2, x3, z1, z2):
        b1 = torch.matmul(x1, x2)
        b2 = torch.mm(x3, z1)
        c = torch.matmul(x3, z2)
        t = b1 - b2 + c
        return t
# Inputs to the model
x1 = torch.randn(5, 5)
x2 = torch.randn(5, 5)
x3 = torch.randn(5, 5)
z1 = torch.randn(5, 5)
z2 = torch.randn(5, 5)
