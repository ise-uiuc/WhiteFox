
class Model(torch.nn.Module):
    def forward(self, x1, x2, x3, x4, x5):
        xx1 = torch.mm(torch.mm(x1, x2), x3)
        xx2 = torch.mm(torch.mm(x4, x5), x3)
        xx3 = xx1 + xx2
        return xx3
# Inputs to the model
x1 = torch.randn(1, 32)
x2 = torch.randn(32, 16)
x3 = torch.randn(16, 8)
x4 = torch.randn(1, 32)
x5 = torch.randn(32, 16)
