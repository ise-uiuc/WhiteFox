
class Model(torch.nn.Module):
    def forward(self, x1, x2, x3, x4, x5, x6, x7):
        v3 = torch.mm(x3, x7)
        v6 = torch.mm(x6, x5)
        v4 = torch.mm(x4, v6)
        v1 = torch.mm(v3, v6)
        v2 = torch.mm(v4, x6)
        return v1 + v2
# Inputs to the model
x1 = torch.randn(2, 2)
x2 = torch.randn(2, 2)
x3 = torch.randn(2, 2)
x4 = torch.randn(2, 2)
x5 = torch.randn(2, 2)
x6 = torch.randn(2, 2)
x7 = torch.randn(2, 2)
