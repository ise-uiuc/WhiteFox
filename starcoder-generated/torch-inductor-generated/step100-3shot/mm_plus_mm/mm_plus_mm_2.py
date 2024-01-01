
class Model(torch.nn.Module):
    def forward(self, a, b, c, d, e):
        g1 = torch.mm(a, b)
        g2 = torch.mm(e, e)
        g = g1 + g2
        g3 = torch.mm(d, e)
        g4 = torch.mm(d, e)
        return g + g3 + g4
# Inputs to the model
a = torch.randn(2, 2, requires_grad=True)
b = torch.randn(2, 2, requires_grad=True)
c = torch.randn(4, 2, requires_grad=True)
d = torch.randn(4, 2, requires_grad=True)
e = torch.randn(4, 4, requires_grad=True)
