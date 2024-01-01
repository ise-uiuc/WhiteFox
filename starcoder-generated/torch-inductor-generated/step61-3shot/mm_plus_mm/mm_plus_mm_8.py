
class Model(torch.nn.Module):
    def forward(self, x1, x2, x3, x4):
        t1 = torch.mm(x1, x2)
        t2 = torch.mm(x2, x3)
        t3 = torch.mm(x4, x1)
        t4 = torch.mm(x1, x3)
        return t1 + t2 + t3 + t4
# Inputs to the model
x1 = torch.randn(16, 16)
x2 = torch.randn(16, 16)
x3 = torch.randn(16, 16)
x4 = torch.randn(16, 16)
