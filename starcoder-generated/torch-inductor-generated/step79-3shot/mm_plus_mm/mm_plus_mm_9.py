
class Model(torch.nn.Module):
    def forward(self, x, y, z, u, v, w):
        t0 = torch.mm(x, y)
        t1 = torch.mm(z, u)
        t2 = torch.mm(v, w)
        t3 = t0 + t1 + t2
        return t3
# Inputs to the model
x = torch.randn(3, 3)
y = torch.randn(3, 3)
z = torch.randn(3, 3)
u = torch.randn(3, 3)
v = torch.randn(3, 3)
w = torch.randn(3, 3)
