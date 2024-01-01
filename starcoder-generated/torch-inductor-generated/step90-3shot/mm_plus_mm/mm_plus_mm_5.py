
class Model(nn.Module):
    def forward(self, x1, x2, x3, x4, x5, x6):
        h1 = torch.mm(x1, x4)
        h2 = torch.mm(x2, x5)
        h3 = torch.mm(x4, x6)
        h4 = torch.mm(x3, x5)
        h5 = torch.mm(x6, x2)
        h6 = torch.mm(x3, x4)
        return (h1 + h4 + h6)
# Inputs to the model
x1 = torch.randn(8, 8)
x2 = torch.randn(8, 8)
x3 = torch.randn(8, 8)
x4 = torch.randn(8, 8)
x5 = torch.randn(8, 8)
x6 = torch.randn(8, 8)
