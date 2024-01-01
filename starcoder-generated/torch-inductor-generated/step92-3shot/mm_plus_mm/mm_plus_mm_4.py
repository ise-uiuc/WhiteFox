
class Model(torch.nn.Module):
    def forward(self, x3, x4, w1, w2, x, y, z):
        w = x + y + z
        t1 = w + w
        t2 = t1 + t1
        t3 = torch.mm(x3, x4)
        t4 = torch.mm(w2, x2) + torch.mm(x3, x4)
        return t3 + t2
# Inputs to the model
w1 = torch.randn(3, 3)
w2 = torch.randn(3, 3)
x = torch.randn(3, 3)
y = torch.randn(2, 2)
z = torch.randn(1, 1)
x1 = torch.randn(3, 3)
x2 = torch.randn(3, 3)
x3 = torch.randn(3, 3)
x4 = torch.randn(3, 3)
