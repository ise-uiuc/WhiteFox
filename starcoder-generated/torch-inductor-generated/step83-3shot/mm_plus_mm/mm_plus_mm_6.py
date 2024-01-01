
class Model(torch.nn.Module):
    def forward(self, x, y, z):
        a = torch.mm(z, y)
        b = torch.mm(x, z)
        c = torch.mm(x, y)
        d = torch.mm(c, a)
        return d
# Inputs to the model
x = torch.randn(4, 6)
y = torch.randn(6, 4)
z = torch.randn(4, 6)
