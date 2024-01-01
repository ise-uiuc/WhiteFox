
class Model(nn.Module):
    def forward(self, x1, x2, x3, x4, x5, x6, x7, x8):
        h = torch.mm(x1, x5) + torch.mm(x2, x6) + torch.mm(x7, x3) + torch.mm(x8, x4)
        return torch.mm(h, h)
# Inputs to the model
x1 = torch.randn(16, 16)
x2 = torch.randn(16, 16)
x3 = torch.randn(16, 16)
x4 = torch.randn(16, 16)
x5 = torch.randn(16, 16)
x6 = torch.randn(16, 16)
x7 = torch.randn(16, 16)
x8 = torch.randn(16, 16)
