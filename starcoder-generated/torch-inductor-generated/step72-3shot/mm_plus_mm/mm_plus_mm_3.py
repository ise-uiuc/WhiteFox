
class Model(nn.Module):
    def forward(self, x1, x2, x3, x4, x5, x6, x7, x8):
        h1 = torch.mm(x1 + x2, x4 + x3)
        h2 = torch.mm(x2, x5 + x6)
        h3 = torch.mm(x3, x7 + x8)
        return h1 + h2 + h3
# Inputs to the model
x1 = torch.randn(4, 4)
x2 = torch.randn(4, 4)
x3 = torch.randn(4, 4)
x4 = torch.randn(4, 4)
x5 = torch.randn(4, 4)
x6 = torch.randn(4, 4)
x7 = torch.randn(4, 4)
x8 = torch.randn(4, 4)
