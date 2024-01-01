
class Model(torch.nn.Module):
    def forward(self, a, b, c, d, e):
        t1 = torch.mm(a, b)
        t2 = torch.mm(t1, t1)
        t1 += t2
        t1 = torch.mm(t1, t1)
        return t1.mm(d)
# Inputs to the model
a = torch.randn(2, 2)
b = torch.randn(2, 2)
c = torch.randn(2, 2)
d = torch.randn(2, 2)
e = torch.randn(2, 2)
