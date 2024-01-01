
class Model(torch.nn.Module):
    def forward(self, a, b, c, d):
        t1 = torch.mm(a, b)
        t2 = torch.mm(c, d)
        t3 = torch.mm(c, a)
        out = t1 + t2 + t3
        return out
# Inputs to the model
a = torch.randn(20, 20)
b = torch.randn(20, 20)
c = torch.randn(20, 20)
d = torch.randn(20, 20)
