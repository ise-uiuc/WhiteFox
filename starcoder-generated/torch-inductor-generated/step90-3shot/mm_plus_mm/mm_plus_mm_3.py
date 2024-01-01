
class Model(nn.Module):
    def forward(self, x1, x2, x3, x4, x5, x6, x7, x8):
        h3 = torch.mm(x1, x3)
        h4 = torch.mm(x4, x5)
        h5 = ((h3 + h4) * (h3 + h4))
        h6 = (x5 * x6)
        h7 = (x7 * x5)
        h8 = (x7 * x6)
        h9 = torch.mm(h5, h6)
        hm = torch.mm(h9, h8)
        return hm.mm(x7 * hm)
# Inputs to the model
x1 = torch.randn(6, 3)
x2 = torch.randn(3, 2)
x3 = torch.randn(6, 2)
x4 = torch.randn(3, 8)
x5 = torch.randn(8, 2)
x6 = torch.randn(6, 2)
x7 = torch.randn(8, 2)
x8 = torch.randn(2, 2)
