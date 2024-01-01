
class Model(torch.nn.Module):
    def forward(self, x, x1, x2, d=1):
        x3 = x1 * x2
        x4 = torch.mm(x, x)
        x5 = x * x
        x6 = x4 - x5
        x7 = torch.mm(x5, x6)
        x8 = torch.mm(x2, x5)
        x9 = x5 ** d
        x10 = x4 * x3
        x11 = x1 - x2
        x12 = torch.mm(x11, x1)
        x13 = torch.mm(x6, x7)
        return x10 - x12 + x13 - x8 - x9 + x9
# Inputs to the model
x1 = torch.randn(12, 10)
x2 = torch.randn(12, 10)
x = torch.randn(10, 10)
