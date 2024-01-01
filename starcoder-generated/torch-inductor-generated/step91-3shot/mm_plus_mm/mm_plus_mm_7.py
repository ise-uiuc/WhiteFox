
class Model(torch.nn.Module):
    def forward(self, x, y, z):
        t1 = torch.mm(y, z)
        t2 = torch.mm(x, x)
        out = t1 + t2 + y
        return out
# Inputs to the model
x = torch.randn(100, 100)
y = torch.randn(100, 100)
z = torch.randn(100, 100)
