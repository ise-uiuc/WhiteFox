
class Model(torch.nn.Module):
    def forward(self, x1, x2, x3, x4, x5, x6):
        B = torch.mm(x1 * x2, x3 * x4)
        A = torch.mm(x5 * x6, x3 * x4) + torch.mm(x5 * x6, x1 * x2)
        B = B + A
        return B + B
# Inputs to the model
x1 = torch.randn(3, 4)
x2 = torch.randn(3, 4)
x3 = torch.randn(3, 4)
x4 = torch.randn(3, 4)
x5 = torch.randn(3, 4)
x6 = torch.randn(3, 4)
