
class Model(torch.nn.Module):
    def forward(self, x1, x2, x3, x4):
        a = torch.mm(x1, x2)
        b = torch.mm(x3, x4)
        c = torch.mm(a, b)
        d = torch.mm(b, x2)
        e = torch.mm(a, b)
        return (a + b + c + d) + e
# Inputs to the model
x1 = torch.randn(5, 5)
x2 = torch.randn(5, 5)
x3 = torch.randn(5, 5)
x4 = torch.randn(5, 5)
